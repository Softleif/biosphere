/// GPU inference correctness: GpuForest::predict must match FlatForest::predict
/// within f32 precision.
///
/// Requires a real GPU. This test will panic if no GPU adapter is available.
#[cfg(feature = "gpu")]
mod gpu_tests {
    #![allow(non_snake_case)]
    use biosphere::gpu::GpuForest;
    use biosphere::{FlatForest, RandomForest, RandomForestParameters};
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_data(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> (Array2<f64>, ndarray::Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let X = Array2::random_using(
            (n_samples, n_features),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        // Use first feature as target to avoid cumsum precision issues during training.
        let y = X.column(0).to_owned();
        (X, y)
    }

    #[test]
    fn gpu_matches_flat_cpu() {
        let n_features = 10;
        let (train_x, train_y) = make_data(150, n_features, 42);

        let params = RandomForestParameters::default()
            .with_n_estimators(20)
            .with_seed(1);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let (test_x, _) = make_data(100, n_features, 99);
        let flat = FlatForest::from_forest(&forest, n_features);
        let test_x_f32 = test_x.mapv(|v| v as f32);
        let cpu_preds = flat.predict(&test_x_f32.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 100);
        let gpu_preds = gpu_forest.predict(&test_x_f32.view());

        // Both FlatForest and GpuForest use f32 nodes and f32 comparisons.
        // The only difference is f64 vs f32 accumulation; tolerance ~1e-5.
        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            let tol = 1e-5;
            assert!(
                ((*cpu as f32) - gpu).abs() < tol,
                "sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    #[test]
    fn gpu_single_tree_single_sample() {
        let n_features = 4;
        let (train_x, train_y) = make_data(100, n_features, 7);

        let params = RandomForestParameters::default()
            .with_n_estimators(1)
            .with_seed(0);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let (test_x, _) = make_data(1, n_features, 55);
        let flat = FlatForest::from_forest(&forest, n_features);
        let test_x_f32 = test_x.mapv(|v| v as f32);
        let cpu_preds = flat.predict(&test_x_f32.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 1);
        let gpu_preds = gpu_forest.predict(&test_x_f32.view());

        assert!(
            ((cpu_preds[0] as f32) - gpu_preds[0]).abs() < 1e-5,
            "cpu={}, gpu={}",
            cpu_preds[0],
            gpu_preds[0]
        );
    }

    // --- Test #8/#9: n_samples at workgroup boundary values ---
    /// Tests 63, 64, 65, 127, 128, 129 samples — exercises the bounds check in the
    /// traverse shader and the exact-workgroup-multiple fast paths.
    #[test]
    fn gpu_workgroup_boundaries() {
        let n_features = 5;
        let (train_x, train_y) = make_data(150, n_features, 42);

        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(1);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        // Allocate enough capacity for the largest batch we'll test.
        let gpu_forest = GpuForest::from_flat_forest(&flat, 200);

        for &n_samples in &[1usize, 63, 64, 65, 127, 128, 129] {
            let (test_x, _) = make_data(n_samples, n_features, n_samples as u64 + 100);
            let test_x_f32 = test_x.mapv(|v| v as f32);

            let cpu_preds = flat.predict(&test_x_f32.view());
            let gpu_preds = gpu_forest.predict(&test_x_f32.view());

            assert_eq!(gpu_preds.len(), n_samples, "n_samples={n_samples}: output length mismatch");
            for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
                assert!(
                    ((*cpu as f32) - gpu).abs() < 1e-5,
                    "n_samples={n_samples} sample {i}: cpu={cpu}, gpu={gpu}"
                );
            }
        }
    }

    // --- Test #10: multiple predict() calls on the same GpuForest ---
    /// Guards against accidental state mutation between calls.
    #[test]
    fn gpu_multiple_predict_calls() {
        let n_features = 6;
        let (train_x, train_y) = make_data(150, n_features, 77);

        let params = RandomForestParameters::default()
            .with_n_estimators(15)
            .with_seed(3);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let gpu_forest = GpuForest::from_flat_forest(&flat, 100);

        let (test_x1, _) = make_data(80, n_features, 10);
        let (test_x2, _) = make_data(60, n_features, 20);
        let test_x1_f32 = test_x1.mapv(|v| v as f32);
        let test_x2_f32 = test_x2.mapv(|v| v as f32);

        let cpu1 = flat.predict(&test_x1_f32.view());
        let cpu2 = flat.predict(&test_x2_f32.view());

        let gpu1 = gpu_forest.predict(&test_x1_f32.view());
        let gpu2 = gpu_forest.predict(&test_x2_f32.view());

        for (i, (cpu, gpu)) in cpu1.iter().zip(gpu1.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "call 1 sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
        for (i, (cpu, gpu)) in cpu2.iter().zip(gpu2.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "call 2 sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }

    // --- Test #11: large n_trees (500) ---
    /// Stresses the y-dispatch dimension and the per-tree-preds buffer size.
    #[test]
    fn gpu_large_n_trees() {
        let n_features = 5;
        let (train_x, train_y) = make_data(150, n_features, 99);

        let params = RandomForestParameters::default()
            .with_n_estimators(500)
            .with_seed(4);
        let mut forest = RandomForest::new(params);
        forest.fit(&train_x.view(), &train_y.view());

        let flat = FlatForest::from_forest(&forest, n_features);
        let (test_x, _) = make_data(50, n_features, 200);
        let test_x_f32 = test_x.mapv(|v| v as f32);

        let cpu_preds = flat.predict(&test_x_f32.view());
        let gpu_forest = GpuForest::from_flat_forest(&flat, 50);
        let gpu_preds = gpu_forest.predict(&test_x_f32.view());

        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            assert!(
                ((*cpu as f32) - gpu).abs() < 1e-5,
                "sample {i}: cpu={cpu}, gpu={gpu}"
            );
        }
    }
}
