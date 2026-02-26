//! Verify that FlatForest::predict produces identical results to RandomForest::predict.
#![allow(non_snake_case)]
use biosphere::{FlatForest, RandomForest, RandomForestParameters};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn make_data(n_samples: usize, n_features: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    // Keep values in [0, 1) to avoid cumsum precision issues in biosphere internals.
    let X = Array2::random_using(
        (n_samples, n_features),
        Uniform::new(0.0, 1.0).unwrap(),
        &mut rng,
    );
    // Simple target: first feature only.
    let y = X.column(0).to_owned();
    (X, y)
}

#[test]
fn flat_forest_matches_recursive() {
    let (X, y) = make_data(200, 10, 42);

    let params = RandomForestParameters::default()
        .with_n_estimators(20)
        .with_seed(1);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(100, 10, 99);
    let cpu_preds = forest.predict(&test_X.view());

    let flat = FlatForest::from_forest(&forest, 10);
    let flat_preds = flat.predict(&test_X.view());

    for (i, (cpu, flat)) in cpu_preds.iter().zip(flat_preds.iter()).enumerate() {
        assert!(
            (cpu - flat).abs() < 1e-12,
            "sample {i}: cpu={cpu}, flat={flat}"
        );
    }
}

#[test]
fn flat_forest_single_sample() {
    let (X, y) = make_data(100, 5, 7);

    let params = RandomForestParameters::default()
        .with_n_estimators(10)
        .with_seed(2);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(1, 5, 55);
    let cpu_preds = forest.predict(&test_X.view());
    let flat = FlatForest::from_forest(&forest, 5);
    let flat_preds = flat.predict(&test_X.view());

    assert!(
        (cpu_preds[0] - flat_preds[0]).abs() < 1e-12,
        "cpu={}, flat={}",
        cpu_preds[0],
        flat_preds[0]
    );
}

#[test]
fn flat_forest_deep_tree() {
    // Trees trained with no max_depth can grow very deep — verify explicit-index layout is correct.
    let (X, y) = make_data(150, 8, 123);

    let params = RandomForestParameters::default()
        .with_n_estimators(5)
        .with_seed(3);
    let mut forest = RandomForest::new(params);
    forest.fit(&X.view(), &y.view());

    let (test_X, _) = make_data(50, 8, 456);
    let cpu_preds = forest.predict(&test_X.view());
    let flat = FlatForest::from_forest(&forest, 8);
    let flat_preds = flat.predict(&test_X.view());

    for (i, (cpu, flat)) in cpu_preds.iter().zip(flat_preds.iter()).enumerate() {
        assert!(
            (cpu - flat).abs() < 1e-12,
            "sample {i}: cpu={cpu}, flat={flat}"
        );
    }
}
