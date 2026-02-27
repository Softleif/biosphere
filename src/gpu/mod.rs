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
//! // 4. Run inference. Features must be f32 (cast from f64 if needed).
//! let test_X: Array2<f64> = /* ... your test data ... */
//! # Array2::zeros((1, 1));
//! let test_X_f32 = test_X.mapv(|v| v as f32);
//! let predictions: ndarray::Array1<f32> = gpu_forest.predict(&test_X_f32.view());
//! ```
//!
//! # Data flow
//!
//! ```text
//! RandomForest  ──from_forest──►  FlatForest (f32 nodes, CPU)
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
//!                           Array1<f32> output
//! ```
//!
//! # Precision
//!
//! [`FlatForest`] stores thresholds and leaf values as `f32`. CPU inference
//! casts feature values to `f32` before comparison so that split decisions are
//! identical to GPU inference. Results may differ slightly from
//! [`RandomForest::predict`] (which uses `f64` throughout); the difference is
//! within normal f32 rounding error (~1 × 10⁻⁷ relative).
//!
//! Because both [`FlatForest::predict`] and [`GpuForest::predict`] use `f32`
//! comparisons, their predictions agree very closely. The only remaining
//! difference is that `FlatForest` accumulates leaf values in `f64` while the
//! GPU shader accumulates in `f32`; this contributes at most ~n_trees × 10⁻⁷
//! additional error.
//!
//! [`FlatForest`]: crate::FlatForest
//! [`RandomForest::predict`]: crate::RandomForest::predict

mod pipeline;
pub use pipeline::{GpuForest, PredictHandle};
