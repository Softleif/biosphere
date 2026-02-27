use crate::flat_forest::{FlatForest, FlatNode, ForestMeta};
use ndarray::{Array1, ArrayView2};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU state shared across all [`GpuForest`] instances forked from the same forest.
///
/// Holds the GPU device, compiled pipelines, and static node/meta buffers.
/// Wrapped in [`Arc`] so [`GpuForest::fork`] avoids re-uploading node data or
/// recompiling shaders when creating per-thread handles.
struct GpuForestShared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    traverse_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    traverse_bgl: wgpu::BindGroupLayout,
    reduce_bgl: wgpu::BindGroupLayout,
    /// Static buffer holding all tree nodes. Never mutated after upload.
    node_buffer: wgpu::Buffer,
    /// Static buffer holding forest metadata. Never mutated after upload.
    meta_buffer: wgpu::Buffer,
    meta: ForestMeta,
    /// Workgroup size chosen from device limits at pipeline-compile time.
    workgroup_size: u32,
    /// True when the adapter supports `MAPPABLE_PRIMARY_BUFFERS` (Apple Silicon,
    /// Vulkan UMA, DX12 UMA). On these devices the CPU and GPU share the same
    /// physical memory, so feature upload and output readback can use direct
    /// buffer mapping instead of staging copies.
    uma: bool,
}

/// A submitted GPU inference job awaiting results.
///
/// Created by [`GpuForest::predict_submit`]. Call [`PredictHandle::collect`] to
/// wait for the GPU to finish and retrieve the predictions.
///
/// Submitting multiple handles before calling `collect` on any of them lets the
/// GPU execute the underlying work concurrently.
pub struct PredictHandle<'forest> {
    forest: &'forest GpuForest,
    submit_idx: wgpu::SubmissionIndex,
    output_bytes: u64,
}

impl<'forest> PredictHandle<'forest> {
    /// Block until the GPU finishes this submission and return the predictions.
    ///
    /// One `f32` per sample, in the same order as the input to [`GpuForest::predict_submit`].
    pub fn collect(self) -> Array1<f32> {
        let device = &self.forest.shared.device;

        if self.forest.shared.uma {
            // UMA path: output_buffer has MAP_READ usage, so we read directly
            // without a staging copy. The poll also waits for the GPU submission.
            let slice = self.forest.output_buffer.slice(..self.output_bytes);
            slice.map_async(wgpu::MapMode::Read, |r| {
                r.expect("GPU output buffer mapping failed");
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(self.submit_idx),
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .expect("GPU poll failed");
            let data = slice.get_mapped_range();
            let result = Array1::from(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
            drop(data);
            self.forest.output_buffer.unmap();
            result
        } else {
            // Discrete GPU path: results were copied to the staging buffer by the
            // command encoder; map that and read back.
            let staging = self.forest.staging_buffer.as_ref().unwrap();
            let buffer_slice = staging.slice(..self.output_bytes);
            buffer_slice.map_async(wgpu::MapMode::Read, |result| {
                result.expect("GPU staging buffer mapping failed");
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(self.submit_idx),
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .expect("GPU poll failed");
            let data = buffer_slice.get_mapped_range();
            let result = Array1::from(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
            drop(data);
            staging.unmap();
            result
        }
    }
}

/// A random forest loaded onto the GPU for batched inference.
///
/// Create with [`GpuForest::from_flat_forest`], then call [`GpuForest::predict`].
///
/// Use [`GpuForest::fork`] to create per-thread handles that share compiled
/// pipelines and uploaded node data without re-uploading or recompiling.
/// Each handle has its own pre-allocated GPU inference buffers and is safe
/// to use from a single thread concurrently with other handles forked from
/// the same forest.
pub struct GpuForest {
    shared: Arc<GpuForestShared>,
    /// Pre-allocated feature buffer.
    /// UMA: `STORAGE | MAP_WRITE` — CPU writes directly via mapping.
    /// Discrete: `STORAGE | COPY_DST` — written via `queue.write_buffer`.
    /// Sized for `max_samples * n_features * 4` bytes.
    feature_buffer: wgpu::Buffer,
    /// Pre-allocated intermediate per-tree predictions (STORAGE).
    /// Sized for `max_samples * n_trees * 4` bytes.
    per_tree_preds_buffer: wgpu::Buffer,
    /// Pre-allocated final per-sample means.
    /// UMA: `STORAGE | MAP_READ` — CPU reads directly; no staging copy needed.
    /// Discrete: `STORAGE | COPY_SRC` — copied to staging buffer before readback.
    /// Sized for `max_samples * 4` bytes.
    output_buffer: wgpu::Buffer,
    /// CPU-readable staging buffer for discrete GPUs. `None` on UMA devices.
    /// Discrete: `COPY_DST | MAP_READ`, sized for `max_samples * 4` bytes.
    staging_buffer: Option<wgpu::Buffer>,
    max_samples: usize,
}

impl GpuForest {
    /// Upload a [`FlatForest`] to the GPU and compile the compute pipelines.
    ///
    /// `max_samples` is the maximum batch size for a single [`predict`] call.
    /// All GPU inference buffers are pre-allocated at this capacity, so individual
    /// `predict` calls avoid any GPU memory allocation overhead.
    ///
    /// On Apple Silicon and other unified-memory adapters, feature upload and
    /// output readback use direct buffer mapping instead of staging copies,
    /// eliminating redundant data movement within the shared memory pool.
    ///
    /// Node thresholds and leaf values are already `f32` in [`FlatForest`], so
    /// no precision loss occurs during upload.
    ///
    /// [`predict`]: GpuForest::predict
    pub fn from_flat_forest(flat: &FlatForest, max_samples: usize) -> Self {
        pollster::block_on(Self::from_flat_forest_async(flat, max_samples))
    }

    async fn from_flat_forest_async(flat: &FlatForest, max_samples: usize) -> Self {
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

        // On unified-memory adapters (Apple Silicon, Vulkan/DX12 UMA) the
        // MAPPABLE_PRIMARY_BUFFERS feature lets us combine STORAGE with MAP_READ
        // / MAP_WRITE. This maps to MTLStorageModeShared on Metal, allowing
        // direct CPU↔GPU access with no staging blit.
        let uma = adapter
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: if uma {
                    wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await
            .expect("Failed to create GPU device");

        // --- Static buffers ---
        // FlatNode is #[repr(C)] + bytemuck::Pod, so we can upload the node slice
        // directly without any conversion.
        let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("biosphere::gpu::nodes"),
            contents: bytemuck::cast_slice::<FlatNode, u8>(&flat.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ForestMeta is #[repr(C)] + bytemuck::Pod; upload directly.
        let meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("biosphere::gpu::meta"),
            contents: bytemuck::bytes_of(&flat.meta),
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

        // --- Workgroup size ---
        // Clamp to 256: larger groups rarely help for memory-bandwidth-bound kernels
        // and can hurt occupancy. Both limits are checked: the per-workgroup cap and
        // the per-dimension cap (relevant for our 1-D (wg_size, 1, 1) dispatch).
        let limits = device.limits();
        let workgroup_size = limits
            .max_compute_invocations_per_workgroup
            .min(limits.max_compute_workgroup_size_x)
            .min(256);
        let wg_constants = [("wg_size", workgroup_size as f64)];

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
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &wg_constants,
                ..Default::default()
            },
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
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &wg_constants,
                ..Default::default()
            },
            cache: None,
        });

        let shared = Arc::new(GpuForestShared {
            device,
            queue,
            traverse_pipeline,
            reduce_pipeline,
            traverse_bgl,
            reduce_bgl,
            node_buffer,
            meta_buffer,
            meta: flat.meta,
            workgroup_size,
            uma,
        });

        Self::alloc_buffers(shared, max_samples)
    }

    /// Allocate the per-instance inference buffers for `max_samples` capacity.
    fn alloc_buffers(shared: Arc<GpuForestShared>, max_samples: usize) -> Self {
        let device = &shared.device;
        let n_trees = shared.meta.n_trees as usize;
        let n_features = shared.meta.n_features as usize;
        let uma = shared.uma;

        let feature_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::features"),
            size: (max_samples * n_features * size_of::<f32>()) as u64,
            usage: if uma {
                // MAP_WRITE: CPU can write directly; Metal allocates as
                // MTLStorageModeShared, so no staging blit is needed.
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE
            } else {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            },
            mapped_at_creation: false,
        });

        let per_tree_preds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::per_tree_preds"),
            size: (max_samples * n_trees * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("biosphere::gpu::output"),
            size: (max_samples * size_of::<f32>()) as u64,
            usage: if uma {
                // MAP_READ: CPU can read directly from the same physical memory
                // the GPU wrote to; no copy_buffer_to_buffer needed.
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ
            } else {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            },
            mapped_at_creation: false,
        });

        let staging_buffer = if uma {
            None
        } else {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("biosphere::gpu::staging"),
                size: (max_samples * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        };

        GpuForest {
            shared,
            feature_buffer,
            per_tree_preds_buffer,
            output_buffer,
            staging_buffer,
            max_samples,
        }
    }

    /// Create a new `GpuForest` handle for use from a separate thread.
    ///
    /// The returned handle shares the compiled GPU pipelines and uploaded node
    /// data with `self` (no recompilation or re-upload). It gets its own
    /// pre-allocated inference buffers sized for `max_samples`, making it safe
    /// to call [`predict`] on from a single thread concurrently with `self` or
    /// any other forked handle.
    ///
    /// [`predict`]: GpuForest::predict
    pub fn fork(&self, max_samples: usize) -> Self {
        Self::alloc_buffers(Arc::clone(&self.shared), max_samples)
    }

    /// Submit a batch inference job to the GPU and return a handle to collect results later.
    ///
    /// The GPU begins work immediately. Call [`PredictHandle::collect`] on the returned handle
    /// to wait for completion and retrieve predictions. Submitting multiple handles before
    /// collecting any allows the GPU to execute the work concurrently.
    ///
    /// Returns `None` when `X` has zero rows.
    ///
    /// `X` is a 2-D array of shape `(n_samples, n_features)`. If `X` is not
    /// already in C (row-major) contiguous order it will be copied before upload.
    ///
    /// # Panics
    ///
    /// Panics if `X.nrows() > max_samples` (the value passed to [`GpuForest::from_flat_forest`])
    /// or if `X.ncols() != n_features`.
    pub fn predict_submit(&self, X: &ArrayView2<f32>) -> Option<PredictHandle<'_>> {
        let n_samples = X.nrows();
        if n_samples == 0 {
            return None;
        }
        assert!(
            n_samples <= self.max_samples,
            "n_samples={n_samples} exceeds max_samples={}",
            self.max_samples
        );
        let n_features = self.shared.meta.n_features as usize;
        assert_eq!(
            X.ncols(),
            n_features,
            "X.ncols()={} must equal n_features={n_features}",
            X.ncols(),
        );

        // Ensure row-major contiguous layout (zero-copy if already standard layout).
        let X_c = X.as_standard_layout();
        let features = X_c
            .as_slice()
            .expect("standard layout is always contiguous");

        let shared = &self.shared;
        let device = &shared.device;
        let queue = &shared.queue;
        let n_trees = shared.meta.n_trees as usize;

        let feature_bytes = (n_samples * n_features * size_of::<f32>()) as u64;
        let per_tree_bytes = (n_samples * n_trees * size_of::<f32>()) as u64;
        let output_bytes = (n_samples * size_of::<f32>()) as u64;

        if shared.uma {
            // UMA path: map feature_buffer directly and write without a staging blit.
            // The buffer is guaranteed to be free here — the caller must have collected
            // (or never started) the previous PredictHandle before calling predict_submit
            // again on the same GpuForest. The poll should return immediately.
            let slice = self.feature_buffer.slice(..feature_bytes);
            slice.map_async(wgpu::MapMode::Write, |r| {
                r.expect("feature buffer map failed");
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("GPU poll for feature buffer map failed");
            {
                let mut mapped = slice.get_mapped_range_mut();
                mapped.copy_from_slice(bytemuck::cast_slice(features));
            }
            self.feature_buffer.unmap();
        } else {
            queue.write_buffer(&self.feature_buffer, 0, bytemuck::cast_slice(features));
        }

        // Sub-range bind groups so that arrayLength() in the shaders reflects
        // the actual n_samples, not the full max_samples capacity.
        let traverse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shared.traverse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shared.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shared.node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.feature_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(feature_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.per_tree_preds_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(per_tree_bytes),
                    }),
                },
            ],
        });

        let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shared.reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.per_tree_preds_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(per_tree_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.output_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(output_bytes),
                    }),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("biosphere::gpu::traverse_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&shared.traverse_pipeline);
            cpass.set_bind_group(0, &traverse_bg, &[]);
            let x_groups = n_samples.div_ceil(shared.workgroup_size as usize) as u32;
            cpass.dispatch_workgroups(x_groups, shared.meta.n_trees, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("biosphere::gpu::reduce_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&shared.reduce_pipeline);
            cpass.set_bind_group(0, &reduce_bg, &[]);
            let x_groups = n_samples.div_ceil(shared.workgroup_size as usize) as u32;
            cpass.dispatch_workgroups(x_groups, 1, 1);
        }

        if !shared.uma {
            // Discrete GPU: copy output to the staging buffer so the CPU can read it.
            encoder.copy_buffer_to_buffer(
                &self.output_buffer,
                0,
                self.staging_buffer.as_ref().unwrap(),
                0,
                output_bytes,
            );
        }

        let submit_idx = queue.submit(Some(encoder.finish()));

        Some(PredictHandle {
            forest: self,
            submit_idx,
            output_bytes,
        })
    }

    /// Run batched inference on `n_samples` samples, blocking until complete.
    ///
    /// Equivalent to `predict_submit(X).map(|h| h.collect()).unwrap_or_default()`.
    /// Use [`GpuForest::predict_submit`] with deferred [`PredictHandle::collect`] to overlap
    /// GPU work across multiple forests.
    ///
    /// `X` is a 2-D array of shape `(n_samples, n_features)`.
    /// Returns one f32 prediction per sample (mean of all tree predictions).
    ///
    /// # Panics
    ///
    /// Panics if `X.nrows() > max_samples` (the value passed to [`GpuForest::from_flat_forest`]).
    pub fn predict(&self, X: &ArrayView2<f32>) -> Array1<f32> {
        self.predict_submit(X)
            .map(|h| h.collect())
            .unwrap_or_else(|| Array1::zeros(0))
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
