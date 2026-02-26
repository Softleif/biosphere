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
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FlatNode {
    pub feature_index: u32,
    pub is_leaf: bool,
    pub threshold: f64,
    pub leaf_value: f64,
    /// Index of the left child within the tree's node slice, or -1 for leaves.
    pub left: i32,
    /// Index of the right child within the tree's node slice, or -1 for leaves.
    pub right: i32,
}

impl FlatNode {
    fn dummy_leaf() -> Self {
        FlatNode {
            feature_index: 0,
            is_leaf: true,
            threshold: 0.0,
            leaf_value: 0.0,
            left: -1,
            right: -1,
        }
    }
}

/// A random forest stored as a flat array of nodes with explicit child indices.
///
/// Each tree occupies a contiguous slice of `max_tree_size` nodes:
/// `nodes[tree_idx * max_tree_size .. (tree_idx + 1) * max_tree_size]`.
///
/// Node 0 of each slice is the root. Internal nodes store explicit `left`/`right`
/// indices (relative to the tree's slice start). Shorter trees are padded with
/// dummy leaf nodes that are never reached during inference.
///
/// This representation enables cache-friendly CPU inference and direct GPU upload
/// without exponential memory blowup for deep, sparse trees.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FlatForest {
    /// Flat node array: length = `n_trees * max_tree_size`.
    pub nodes: Vec<FlatNode>,
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
    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = X.nrows();
        let mut output = Array1::<f64>::zeros(n_samples);

        for sample in 0..n_samples {
            let row = X.row(sample);
            let features = row.as_slice().expect("feature row must be contiguous");

            let mut sum = 0.0f64;
            for tree_idx in 0..self.n_trees {
                sum += self.predict_one(tree_idx, features);
            }
            output[sample] = sum / self.n_trees as f64;
        }

        output
    }

    fn predict_one(&self, tree_idx: usize, features: &[f64]) -> f64 {
        let offset = tree_idx * self.max_tree_size;
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[offset + idx];
            if node.is_leaf {
                return node.leaf_value;
            }
            if features[node.feature_index as usize] < node.threshold {
                idx = node.left as usize;
            } else {
                idx = node.right as usize;
            }
        }
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
                feature_index: 0,
                is_leaf: true,
                threshold: 0.0,
                leaf_value: node.label.unwrap(),
                left: -1,
                right: -1,
            };
        } else {
            let left_idx = next_slot;
            next_slot += 1;
            let right_idx = next_slot;
            next_slot += 1;

            nodes[idx] = FlatNode {
                feature_index: node.feature_index.unwrap() as u32,
                is_leaf: false,
                threshold: node.feature_value.unwrap(),
                leaf_value: 0.0,
                left: left_idx as i32,
                right: right_idx as i32,
            };

            queue.push_back((node.left_child.as_ref().unwrap(), left_idx));
            queue.push_back((node.right_child.as_ref().unwrap(), right_idx));
        }
    }
}
