use crate::forest::RandomForest;
use crate::tree::DecisionTreeNode;
use ndarray::{Array1, ArrayView2};

/// A single node in a BFS-ordered flat tree.
///
/// Internal nodes use `feature_index` and `threshold`; leaf nodes use `leaf_value`.
/// The BFS layout: node at index `i` has left child at `2i + 1`, right child at `2i + 2`.
#[derive(Clone, Debug)]
pub struct FlatNode {
    pub feature_index: u32,
    pub is_leaf: bool,
    pub threshold: f64,
    pub leaf_value: f64,
}

impl FlatNode {
    fn dummy_leaf() -> Self {
        FlatNode {
            feature_index: 0,
            is_leaf: true,
            threshold: 0.0,
            leaf_value: 0.0,
        }
    }
}

/// A random forest stored as a flat BFS-ordered array of nodes.
///
/// All trees are padded to `max_tree_size` nodes so any tree can be indexed as
/// `nodes[tree_idx * max_tree_size + node_idx]`.
///
/// This representation enables both cache-friendly CPU inference and direct GPU upload.
#[derive(Clone, Debug)]
pub struct FlatForest {
    /// Flat node array: length = `n_trees * max_tree_size`.
    pub nodes: Vec<FlatNode>,
    pub n_trees: usize,
    pub n_features: usize,
    /// Nodes per tree (all trees padded to this size).
    pub max_tree_size: usize,
    /// Maximum depth across all trees.
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

        // Compute the max depth across all trees.
        let max_depth = trees
            .iter()
            .map(|t| node_depth(t.root()))
            .max()
            .unwrap_or(0);

        // BFS-complete tree needs 2^(depth+1) - 1 nodes.
        let max_tree_size = (1usize << (max_depth + 1)).saturating_sub(1);

        let mut nodes = vec![FlatNode::dummy_leaf(); n_trees * max_tree_size];

        for (tree_idx, tree) in trees.iter().enumerate() {
            let offset = tree_idx * max_tree_size;
            fill_bfs(tree.root(), &mut nodes[offset..offset + max_tree_size], 0);
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
                idx = 2 * idx + 1; // left
            } else {
                idx = 2 * idx + 2; // right
            }
        }
    }
}

/// Recursively compute the depth of a node (0 = leaf, 1 = one level of splits, …).
fn node_depth(node: &DecisionTreeNode) -> usize {
    if node.feature_index.is_none() {
        0
    } else {
        1 + node_depth(node.left_child.as_ref().unwrap())
            .max(node_depth(node.right_child.as_ref().unwrap()))
    }
}

/// Fill BFS-ordered `nodes` slice starting at `index` by recursively visiting `node`.
fn fill_bfs(node: &DecisionTreeNode, nodes: &mut [FlatNode], index: usize) {
    if index >= nodes.len() {
        return;
    }
    if node.feature_index.is_none() {
        nodes[index] = FlatNode {
            feature_index: 0,
            is_leaf: true,
            threshold: 0.0,
            leaf_value: node.label.unwrap(),
        };
    } else {
        nodes[index] = FlatNode {
            feature_index: node.feature_index.unwrap() as u32,
            is_leaf: false,
            threshold: node.feature_value.unwrap(),
            leaf_value: 0.0,
        };
        fill_bfs(node.left_child.as_ref().unwrap(), nodes, 2 * index + 1);
        fill_bfs(node.right_child.as_ref().unwrap(), nodes, 2 * index + 2);
    }
}
