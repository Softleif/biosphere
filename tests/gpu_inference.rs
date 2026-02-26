/// GPU inference correctness: GpuForest::predict must match FlatForest::predict
/// within f32 precision.
///
/// Requires a real GPU. This test will panic if no GPU adapter is available.
#[cfg(feature = "gpu")]
mod gpu_tests {
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
        let cpu_preds = flat.predict(&test_x.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 100);
        // Convert test features to f32 row-major.
        let features_f32: Vec<f32> = test_x.iter().map(|&v| v as f32).collect();
        let gpu_preds = gpu_forest.predict(&features_f32, 100);

        // GPU uses f32; allow tolerance for f64→f32 conversion.
        for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            let tol = 1e-4;
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
        let cpu_preds = flat.predict(&test_x.view());

        let gpu_forest = GpuForest::from_flat_forest(&flat, 1);
        let features_f32: Vec<f32> = test_x.iter().map(|&v| v as f32).collect();
        let gpu_preds = gpu_forest.predict(&features_f32, 1);

        assert!(
            ((cpu_preds[0] as f32) - gpu_preds[0]).abs() < 1e-4,
            "cpu={}, gpu={}",
            cpu_preds[0],
            gpu_preds[0]
        );
    }
}
