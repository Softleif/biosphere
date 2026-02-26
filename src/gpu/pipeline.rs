use crate::flat_forest::FlatForest;
use wgpu::util::DeviceExt;

/// GPU representation of a single node (f32/i32, 24 bytes, bytemuck-compatible).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNode {
    feature_index: u32,
    is_leaf: u32,
    threshold: f32,
    leaf_value: f32,
    /// Left child index within the tree's node slice, or -1 for leaves.
    left: i32,
    /// Right child index within the tree's node slice, or -1 for leaves.
    right: i32,
}

/// Forest metadata passed to every shader invocation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ForestMeta {
    n_trees: u32,
    n_features: u32,
    max_tree_size: u32,
    max_depth: u32,
}

/// A random forest loaded onto the GPU for batched inference.
///
/// Create with [`GpuForest::from_flat_forest`], then call [`GpuForest::predict`].
pub struct GpuForest {
    device: wgpu::Device,
    queue: wgpu::Queue,
    traverse_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    traverse_bgl: wgpu::BindGroupLayout,
    reduce_bgl: wgpu::BindGroupLayout,
    /// Static buffer holding all tree nodes (f32).
    node_buffer: wgpu::Buffer,
    /// Static buffer holding forest metadata.
    meta_buffer: wgpu::Buffer,
    n_trees: u32,
}

impl GpuForest {
    /// Upload a [`FlatForest`] to the GPU and compile the compute pipelines.
    ///
    /// Node thresholds and leaf values are downcast from f64 to f32.
    /// This may introduce small numerical differences compared to CPU inference
    /// for samples whose feature value is very close to a split threshold.
    pub fn from_flat_forest(flat: &FlatForest) -> Self {
        pollster::block_on(Self::from_flat_forest_async(flat))
    }

    async fn from_flat_forest_async(flat: &FlatForest) -> Self {
        // --- Device initialisation ---
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create GPU device");

        // --- Static buffers ---
        let gpu_nodes: Vec<GpuNode> = flat
            .nodes
            .iter()
            .map(|n| GpuNode {
                feature_index: n.feature_index,
                is_leaf: n.is_leaf as u32,
                threshold: n.threshold as f32,
                leaf_value: n.leaf_value as f32,
                left: n.left,
                right: n.right,
            })
            .collect();

        let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("biosphere::gpu::nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let meta = ForestMeta {
            n_trees: flat.n_trees as u32,
            n_features: flat.n_features as u32,
            max_tree_size: flat.max_tree_size as u32,
            max_depth: flat.max_depth as u32,
        };
        let meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("biosphere::gpu::meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // --- Bind group layouts ---
        let traverse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("biosphere::gpu::traverse_bgl"),
            entries: &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
            ],
        });

        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("biosphere::gpu::reduce_bgl"),
            entries: &[storage_ro_entry(0), storage_rw_entry(1)],
        });

        // --- Shader modules ---
        let traverse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("biosphere::gpu::traverse"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/traverse.wgsl").into()),
        });

        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("biosphere::gpu::reduce"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce.wgsl").into()),
        });

        // --- Compute pipelines ---
        let traverse_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("biosphere::gpu::traverse_layout"),
            bind_group_layouts: &[&traverse_bgl],
            immediate_size: 0,
        });

        let traverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("biosphere::gpu::traverse_pipeline"),
            layout: Some(&traverse_layout),
            module: &traverse_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reduce_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("biosphere::gpu::reduce_layout"),
            bind_group_layouts: &[&reduce_bgl],
            immediate_size: 0,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("biosphere::gpu::reduce_pipeline"),
            layout: Some(&reduce_layout),
            module: &reduce_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        GpuForest {
            device,
            queue,
            traverse_pipeline,
            reduce_pipeline,
            traverse_bgl,
            reduce_bgl,
            node_buffer,
            meta_buffer,
            n_trees: flat.n_trees as u32,
        }
    }

    /// Run batched inference on `n_samples` samples.
    ///
    /// `features` is a row-major f32 matrix of shape `(n_samples, n_features)`.
    /// Returns one f32 prediction per sample (mean of all tree predictions).
    pub fn predict(&self, features: &[f32], n_samples: usize) -> Vec<f32> {
        pollster::block_on(self.predict_async(features, n_samples))
    }

    async fn predict_async(&self, features: &[f32], n_samples: usize) -> Vec<f32> {
        let device = &self.device;
        let queue = &self.queue;

        // --- Per-call buffers ---
        let feature_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("biosphere::gpu::features"),
            contents: bytemuck::cast_slice(features),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let per_tree_preds_size = (n_samples * self.n_trees as usize * size_of::<f32>()) as u64;
        let per_tree_preds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::per_tree_preds"),
            size: per_tree_preds_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_size = (n_samples * size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::staging"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // --- Bind groups ---
        let traverse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("biosphere::gpu::traverse_bg"),
            layout: &self.traverse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: feature_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: per_tree_preds_buffer.as_entire_binding(),
                },
            ],
        });

        let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("biosphere::gpu::reduce_bg"),
            layout: &self.reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: per_tree_preds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Encode and submit ---
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("biosphere::gpu::traverse_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.traverse_pipeline);
            cpass.set_bind_group(0, &traverse_bg, &[]);
            let x_groups = n_samples.div_ceil(64) as u32;
            cpass.dispatch_workgroups(x_groups, self.n_trees, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("biosphere::gpu::reduce_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.reduce_pipeline);
            cpass.set_bind_group(0, &reduce_bg, &[]);
            let x_groups = n_samples.div_ceil(64) as u32;
            cpass.dispatch_workgroups(x_groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        // --- Read results ---
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");

        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    }
}

// --- Helpers ---

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
