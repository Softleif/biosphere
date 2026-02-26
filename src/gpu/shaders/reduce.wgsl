// Reduction kernel: average per-tree predictions into final per-sample output.
//
// Dispatch: (ceil(n_samples / 64), 1, 1)
// Workgroup size: (64, 1, 1)

// per_tree_preds: row-major, shape (n_samples, n_trees)
@group(0) @binding(0) var<storage, read> per_tree_preds: array<f32>;
// output: shape (n_samples,)
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_samples = arrayLength(&output);
    let n_trees = arrayLength(&per_tree_preds) / n_samples;
    let sample_id = gid.x;

    if sample_id >= n_samples {
        return;
    }

    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < n_trees; t = t + 1u) {
        sum = sum + per_tree_preds[sample_id * n_trees + t];
    }
    output[sample_id] = sum / f32(n_trees);
}
