// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use flat_forest::{FlatForest, FlatNode};
pub use forest::RandomForest;
pub use forest::RandomForestParameters;
pub use tree::{DecisionTree, DecisionTreeParameters, MaxFeatures};
mod flat_forest;
mod forest;
mod tree;
pub mod utils; // This is public for benchmarking only.

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(test)]
mod testing;
