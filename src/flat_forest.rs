use crate::forest::RandomForest;
use crate::tree::DecisionTreeNode;
use ndarray::{Array1, ArrayView2};
use std::collections::VecDeque;

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
/// and without the 3-byte padding hole that `bool` caused before `threshold`.
///
/// NOTE: field order is significant for binary serde (e.g. postcard). If you
/// need binary compatibility with data serialised before this change, a
/// migration step is required.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FlatNode {
    /// Index of the left child within the tree's node slice, or -1 for leaves.
    pub left: i32,
    /// Index of the right child within the tree's node slice, or -1 for leaves.
    pub right: i32,
    pub feature_index: u32,
    pub threshold: f64,
    pub leaf_value: f64,
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
/// let predictions = flat.predict(&X.view()); // identical to forest.predict()
/// ```
///
/// Internally, each tree is stored in BFS order with explicit child indices so
/// deep, sparse trees (common in practice) don't require exponential padding.
///
/// [`GpuForest::from_flat_forest`]: crate::gpu::GpuForest::from_flat_forest
/// [`RandomForest`]: crate::RandomForest
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FlatForest {
    /// Flat node array: length = `n_trees * max_tree_size`.
    pub(crate) nodes: Vec<FlatNode>,
    pub n_trees: usize,
    pub n_features: usize,
    /// Nodes per tree (all trees padded to this size).
    pub max_tree_size: usize,
    /// Maximum depth across all trees (used as a GPU traversal bound).
    pub max_depth: usize,
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
            n_trees,
            n_features,
            max_tree_size,
            max_depth,
        }
    }

    /// Run inference on a batch of samples.
    ///
    /// Produces the same results as `RandomForest::predict` (mean of per-tree predictions).
    ///
    /// The loop order is outer=tree, inner=sample so that each tree's node array
    /// stays warm in L3 cache while all samples are processed against it, rather
    /// than re-loading every tree's nodes for every sample.
    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = X.nrows();
        let mut output = Array1::<f64>::zeros(n_samples);

        for tree_idx in 0..self.n_trees {
            for sample in 0..n_samples {
                let row = X.row(sample);
                let features = row.as_slice().expect("feature row must be contiguous");
                output[sample] += self.predict_one(tree_idx, features);
            }
        }

        output / self.n_trees as f64
    }

    fn predict_one(&self, tree_idx: usize, features: &[f64]) -> f64 {
        let offset = tree_idx * self.max_tree_size;
        let mut idx = 0usize;
        for _ in 0..self.max_tree_size {
            let node = &self.nodes[offset + idx];
            if node.left < 0 {
                return node.leaf_value;
            }
            if features[node.feature_index as usize] < node.threshold {
                idx = node.left as usize;
            } else {
                idx = node.right as usize;
            }
        }
        panic!(
            "predict_one: traversal exceeded max_tree_size ({}); tree may be malformed",
            self.max_tree_size
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
                leaf_value: node.label.unwrap(),
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
                threshold: node.feature_value.unwrap(),
                leaf_value: 0.0,
            };

            queue.push_back((node.left_child.as_ref().unwrap(), left_idx));
            queue.push_back((node.right_child.as_ref().unwrap(), right_idx));
        }
    }
}
