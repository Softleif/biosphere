use crate::forest::RandomForest;
use crate::tree::DecisionTreeNode;
use ndarray::{Array1, ArrayView2};
use std::collections::VecDeque;

/// Forest metadata: dimensions shared by CPU and GPU inference.
///
/// NOTE: field order and types are significant for binary serde (e.g. postcard)
/// and for direct GPU upload via bytemuck. Changing them requires a migration step
/// for any data serialised with an earlier layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct ForestMeta {
    pub n_trees: u32,
    pub n_features: u32,
    /// Nodes per tree (all trees are padded to this size).
    pub max_tree_size: u32,
    /// Maximum depth across all trees (used as a GPU traversal bound).
    pub max_depth: u32,
}

/// A single node in a flat tree stored in BFS visit order.
///
/// Internal nodes use `feature_index`, `threshold`, `left`, and `right`.
/// Leaf nodes use `leaf_value`; `left` and `right` are -1.
///
/// Child indices are explicit offsets into the per-tree node slice, so trees
/// of any shape can be stored without exponential padding.
///
/// Field order: traversal-hot fields (`left`, `right`, `feature_index`,
/// `threshold`) come first, the leaf-only field (`leaf_value`) last.
/// `is_leaf` was removed — `left < 0` encodes leaf status without redundancy
/// and without the 4-byte overhead that a separate boolean flag would require.
///
/// All floating-point values use `f32`, matching the GPU representation.
/// CPU `predict_one` casts feature values to `f32` before comparison so that
/// the CPU and GPU traversal paths are identical.
///
/// NOTE: field order and types are significant for binary serde (e.g. postcard)
/// and for direct GPU upload via bytemuck. Changing them requires a migration step
/// for any data serialised with an earlier layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct FlatNode {
    /// Index of the left child within the tree's node slice, or -1 for leaves.
    pub left: i32,
    /// Index of the right child within the tree's node slice, or -1 for leaves.
    pub right: i32,
    pub feature_index: u32,
    pub threshold: f32,
    pub leaf_value: f32,
}

impl FlatNode {
    fn dummy_leaf() -> Self {
        FlatNode {
            left: -1,
            right: -1,
            feature_index: 0,
            threshold: 0.0,
            leaf_value: 0.0,
        }
    }
}

/// A [`RandomForest`] converted to a flat, contiguous array for fast inference.
///
/// This is the bridge between a trained forest and GPU execution. Convert once
/// with [`FlatForest::from_forest`], then either call [`FlatForest::predict`]
/// for CPU inference or pass it to [`GpuForest::from_flat_forest`] to run on
/// the GPU.
///
/// ```rust
/// use biosphere::{FlatForest, RandomForest, RandomForestParameters};
/// use ndarray::array;
///
/// let X = array![[0.0, 1.0], [1.0, 0.0]];
/// let y = array![0.0, 1.0];
/// let mut forest = RandomForest::new(RandomForestParameters::default());
/// forest.fit(&X.view(), &y.view());
///
/// let flat = FlatForest::from_forest(&forest, X.ncols());
/// let predictions = flat.predict(&X.view()); // comparable to forest.predict()
/// ```
///
/// Internally, each tree is stored in BFS order with explicit child indices so
/// deep, sparse trees (common in practice) don't require exponential padding.
///
/// All thresholds and leaf values are stored as `f32`. CPU inference casts
/// feature values to `f32` before comparison so that split decisions are
/// identical to GPU inference. Results are accumulated in `f64` and returned
/// as `f64` for API consistency with [`RandomForest::predict`].
///
/// [`GpuForest::from_flat_forest`]: crate::gpu::GpuForest::from_flat_forest
/// [`RandomForest`]: crate::RandomForest
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FlatForest {
    /// Flat node array: length = `meta.n_trees * meta.max_tree_size`.
    pub(crate) nodes: Vec<FlatNode>,
    pub meta: ForestMeta,
}

impl std::ops::Deref for FlatForest {
    type Target = ForestMeta;
    fn deref(&self) -> &ForestMeta {
        &self.meta
    }
}

impl FlatForest {
    /// Build a `FlatForest` from a trained `RandomForest`.
    ///
    /// `n_features` must match the number of features the forest was trained on.
    pub fn from_forest(forest: &RandomForest, n_features: usize) -> Self {
        let trees = forest.trees();
        let n_trees = trees.len();
        assert!(n_trees > 0, "Forest must have at least one tree");

        let max_depth = trees
            .iter()
            .map(|t| node_depth(t.root()))
            .max()
            .unwrap_or(0);

        // Allocate only as many slots as the largest tree actually needs.
        let max_tree_size = trees
            .iter()
            .map(|t| count_nodes(t.root()))
            .max()
            .unwrap_or(1);

        let mut nodes = vec![FlatNode::dummy_leaf(); n_trees * max_tree_size];

        for (tree_idx, tree) in trees.iter().enumerate() {
            let offset = tree_idx * max_tree_size;
            fill_bfs(tree.root(), &mut nodes[offset..offset + max_tree_size]);
        }

        FlatForest {
            nodes,
            meta: ForestMeta {
                n_trees: n_trees as u32,
                n_features: n_features as u32,
                max_tree_size: max_tree_size as u32,
                max_depth: max_depth as u32,
            },
        }
    }

    /// Run inference on a batch of samples.
    ///
    /// Feature values are cast to `f32` internally so that split decisions are
    /// identical to GPU inference. The per-tree leaf values (f32) are accumulated
    /// in f64 and the final mean is returned as `Array1<f64>`.
    ///
    /// The loop order is outer=tree, inner=sample so that each tree's node array
    /// stays warm in L3 cache while all samples are processed against it, rather
    /// than re-loading every tree's nodes for every sample.
    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = X.nrows();
        let mut output = Array1::<f64>::zeros(n_samples);

        for tree_idx in 0..self.meta.n_trees as usize {
            for sample in 0..n_samples {
                let row = X.row(sample);
                let features = row.as_slice().expect("feature row must be contiguous");
                output[sample] += self.predict_one(tree_idx, features);
            }
        }

        output / self.meta.n_trees as f64
    }

    fn predict_one(&self, tree_idx: usize, features: &[f64]) -> f64 {
        let offset = tree_idx * self.meta.max_tree_size as usize;
        let mut idx = 0usize;
        for _ in 0..self.meta.max_tree_size as usize {
            let node = &self.nodes[offset + idx];
            if node.left < 0 {
                return node.leaf_value as f64;
            }
            // Cast feature to f32 to match GPU traversal exactly.
            if (features[node.feature_index as usize] as f32) < node.threshold {
                idx = node.left as usize;
            } else {
                idx = node.right as usize;
            }
        }
        panic!(
            "predict_one: traversal exceeded max_tree_size ({}); tree may be malformed",
            self.meta.max_tree_size
        );
    }
}

/// Recursively compute the depth of a node (0 = leaf, 1 = one split level, …).
fn node_depth(node: &DecisionTreeNode) -> usize {
    if node.feature_index.is_none() {
        0
    } else {
        1 + node_depth(node.left_child.as_ref().unwrap())
            .max(node_depth(node.right_child.as_ref().unwrap()))
    }
}

/// Count the total number of nodes in the subtree rooted at `node`.
fn count_nodes(node: &DecisionTreeNode) -> usize {
    if node.feature_index.is_none() {
        1
    } else {
        1 + count_nodes(node.left_child.as_ref().unwrap())
            + count_nodes(node.right_child.as_ref().unwrap())
    }
}

/// Fill `nodes` with the tree rooted at `root` in BFS order, storing explicit
/// child indices in each internal node.
fn fill_bfs(root: &DecisionTreeNode, nodes: &mut [FlatNode]) {
    let mut queue: VecDeque<(&DecisionTreeNode, usize)> = VecDeque::new();
    queue.push_back((root, 0));
    let mut next_slot = 1usize;

    while let Some((node, idx)) = queue.pop_front() {
        if node.feature_index.is_none() {
            nodes[idx] = FlatNode {
                left: -1,
                right: -1,
                feature_index: 0,
                threshold: 0.0,
                leaf_value: node.label.unwrap() as f32,
            };
        } else {
            let left_idx = next_slot;
            next_slot += 1;
            let right_idx = next_slot;
            next_slot += 1;

            nodes[idx] = FlatNode {
                left: left_idx as i32,
                right: right_idx as i32,
                feature_index: node.feature_index.unwrap() as u32,
                threshold: node.feature_value.unwrap() as f32,
                leaf_value: 0.0,
            };

            queue.push_back((node.left_child.as_ref().unwrap(), left_idx));
            queue.push_back((node.right_child.as_ref().unwrap(), right_idx));
        }
    }
}
