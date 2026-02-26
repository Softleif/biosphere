// Tree traversal kernel.
//
// Each GPU thread handles one (sample, tree) pair, walking the flat node array
// for that tree to produce a per-tree prediction for that sample.
//
// Nodes are stored in BFS visit order with explicit left/right child indices,
// so trees of any depth and shape are supported without exponential padding.
//
// Dispatch: (ceil(n_samples / 64), n_trees, 1)
// Workgroup size: (64, 1, 1)

struct Node {
    feature_index: u32,
    is_leaf: u32,
    threshold: f32,
    leaf_value: f32,
    left: i32,
    right: i32,
}

struct Meta {
    n_trees: u32,
    n_features: u32,
    max_tree_size: u32,
    max_depth: u32,
}

@group(0) @binding(0) var<storage, read> forest_meta: Meta;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
// features: row-major, shape (n_samples, n_features)
@group(0) @binding(2) var<storage, read> features: array<f32>;
// per_tree_preds: column-major, shape (n_trees, n_samples)
@group(0) @binding(3) var<storage, read_write> per_tree_preds: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_samples = arrayLength(&features) / forest_meta.n_features;
    let sample_id = gid.x;
    let tree_id = gid.y;

    if sample_id >= n_samples {
        return;
    }

    let tree_offset = tree_id * forest_meta.max_tree_size;
    var node_idx: u32 = 0u;

    // Traverse at most max_depth + 1 steps as a safety bound.
    for (var i: u32 = 0u; i <= forest_meta.max_depth; i = i + 1u) {
        let node = nodes[tree_offset + node_idx];
        if node.is_leaf == 1u {
            per_tree_preds[tree_id * n_samples + sample_id] = node.leaf_value;
            return;
        }
        let feat_val = features[sample_id * forest_meta.n_features + node.feature_index];
        // Use strict < to match biosphere's CPU inference exactly.
        if feat_val < node.threshold {
            node_idx = u32(node.left);
        } else {
            node_idx = u32(node.right);
        }
    }
    // Fallback: should not reach here for well-formed trees.
    per_tree_preds[tree_id * n_samples + sample_id] = nodes[tree_offset + node_idx].leaf_value;
}
