Biosphere is a Rust library for fast and simple Random Forests.

# Project Overview

The repository consists of:
- **Rust core library** (`src/`): Core random forest and decision tree implementation
- **Python bindings** (`biosphere-py/`): PyO3-based Python package built with Maturin

## Development Commands

```bash
cargo test --features serde
```

*Note:** Add/update tests when APIs are added or changed.

## Core Components

1. **DecisionTree** (`src/tree/decision_tree.rs`)
   - Individual decision tree implementation
   - Uses sorted samples for efficient splitting
   - Configured via `DecisionTreeParameters`

2. **RandomForest** (`src/forest.rs`)
   - Ensemble of decision trees
   - Parallel training using rayon threadpool
   - OOB (out-of-bag) prediction support
   - Configured via `RandomForestParameters`

3. **Tree Node Structure** (`src/tree/decision_tree_node.rs`)
   - Recursive tree node implementation
   - Handles splitting logic and prediction

4. **Utilities** (`src/utils.rs`)
   - `argsort`: Sorting indices computation
   - `sample_weights`: Bootstrap sample generation
   - `sorted_samples`: Pre-sorting optimization for training

## Important Implementation Notes

**Array Naming Convention:**
The codebase uses capital `X` for feature arrays (following scikit-learn convention) and allows this via `#![allow(non_snake_case)]` in lib.rs.

**Random State:**
- Seeds are u64 values
- Each tree in a forest gets a derived seed for reproducibility
- Uses `StdRng` from rand crate

**MaxFeatures Options:**
The `MaxFeatures` enum supports multiple feature sampling strategies:
- `None`: Use all features
- `Value(usize)`: Use specific number of features
- `Fraction(f64)`: Use fraction of features
- `Sqrt`: Use sqrt(n_features)

## GPU Feature (`--features gpu`)

**What was built:**
- `src/flat_forest.rs` — `FlatForest` / `FlatNode`: flat tree array usable for both CPU and GPU inference. Nodes are stored in BFS visit order with **explicit child indices** (`left: i32`, `right: i32`; -1 for leaves). `max_tree_size` is the actual maximum node count across trees (NOT `2^(max_depth+1) - 1`). `FlatForest::from_forest(forest, n_features)` flattens all trees; `FlatForest::predict` matches `RandomForest::predict` exactly.
- `src/gpu/pipeline.rs` — `GpuForest`: uploads `FlatForest` to GPU (f64→f32), compiles WGSL pipelines, runs batched inference via `predict(&[f32], n_samples) -> Vec<f32>`.
- `src/gpu/shaders/traverse.wgsl` — 2D dispatch `(ceil(n_samples/64), n_trees, 1)`, one thread per (sample, tree). Uses `u32(node.left)` / `u32(node.right)` for traversal; `max_depth` is kept as a GPU loop safety bound only.
- `src/gpu/shaders/reduce.wgsl` — averages per-tree predictions per sample.
- Tests: `tests/flat_forest.rs` (CPU, no feature flag), `tests/gpu_inference.rs` (requires GPU, `--features gpu`).

**Why not BFS positional encoding (`2i+1`/`2i+2`):**
Real-world RF trees trained without a `max_depth` constraint are deep and sparse. A positional BFS layout requires `2^(depth+1) - 1` slots per tree — depth ~44 means petabytes of allocation. Explicit child indices allocate only actual nodes.

## Known pre-existing issue

The debug assertion in `decision_tree_node.rs:180` fires for random training sets > ~300 samples — keep test training sizes ≤ 200 samples.

# Memory

**Keep this file updated.** After every session where you discover something stable and non-obvious about this project, add it below. Remove or correct entries that turn out to be wrong or outdated. Do not duplicate entries already in CLAUDE.md.

# Interactivity guidelines

When you are asked to implement something, always ask for clarifications if needed.
If you are unsure about the requirements, ask for more details.
If you think there is a better way to implement something, suggest it and explain your reasoning, but don't implement it immediately without approval.
