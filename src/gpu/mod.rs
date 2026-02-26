//! GPU-accelerated inference for trained random forests.
//!
//! Enabled with the `gpu` feature flag. Uses [wgpu] compute shaders (Vulkan,
//! Metal, DX12, WebGPU) with no platform-specific code.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use biosphere::{FlatForest, RandomForest, RandomForestParameters};
//! use biosphere::gpu::GpuForest;
//! use ndarray::Array2;
//!
//! // 1. Train a forest as usual.
//! let X: Array2<f64> = /* ... your training data ... */
//! # Array2::zeros((1, 1));
//! let y = /* ... your labels ... */
//! # ndarray::Array1::zeros(1);
//! let mut forest = RandomForest::new(RandomForestParameters::default());
//! forest.fit(&X.view(), &y.view());
//!
//! // 2. Convert to a flat BFS representation (works on CPU too).
//! let n_features = X.ncols();
//! let flat = FlatForest::from_forest(&forest, n_features);
//!
//! // 3. Upload to the GPU, reserving capacity for up to 1024 samples per call.
//! let gpu_forest = GpuForest::from_flat_forest(&flat, 1024);
//!
//! // 4. Run inference. Features must be f32, row-major (n_samples × n_features).
//! let test_X: Array2<f64> = /* ... your test data ... */
//! # Array2::zeros((1, 1));
//! let n_samples = test_X.nrows();
//! let features_f32: Vec<f32> = test_X.iter().map(|&v| v as f32).collect();
//! let predictions: Vec<f32> = gpu_forest.predict(&features_f32, n_samples);
//! ```
//!
//! # Data flow
//!
//! ```text
//! RandomForest  ──from_forest──►  FlatForest (f64, CPU)
//!                                      │
//!                              from_flat_forest
//!                                      │
//!                                      ▼
//!                               GpuForest (f32, GPU)
//!                                      │
//!                                  predict()
//!                                      │
//!                        ┌────────────────────────┐
//!                        │ traverse.wgsl          │
//!                        │  one thread per        │
//!                        │  (sample, tree) pair   │
//!                        └──────────┬─────────────┘
//!                                   │ per-tree predictions
//!                        ┌──────────▼─────────────┐
//!                        │ reduce.wgsl             │
//!                        │  mean across trees      │
//!                        └──────────┬─────────────┘
//!                                   │
//!                              Vec<f32> output
//! ```
//!
//! # Precision
//!
//! [`FlatForest`] stores thresholds and leaf values as `f64` and produces
//! results identical to [`RandomForest::predict`].
//!
//! [`GpuForest`] converts those values to `f32` on upload. For samples whose
//! feature value falls very close to a split threshold the GPU prediction may
//! differ slightly from the CPU prediction. In practice the difference is
//! within normal f32 rounding error (~1 × 10⁻⁷ relative).
//!
//! [`FlatForest`]: crate::FlatForest
//! [`RandomForest::predict`]: crate::RandomForest::predict

mod pipeline;
pub use pipeline::GpuForest;
