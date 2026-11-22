//! Compute pipelines, and the one uniform block they all take.

use bytemuck::{Pod, Zeroable};

use crate::Device;

/// Threads per workgroup, and the number every kernel in this crate declares.
///
/// 64 rather than 256 or 1024. WebGPU's guaranteed floor is 256 invocations
/// per workgroup, so 64 is portable by a wide margin; it is also a whole
/// number of Metal SIMD-groups (32) and of AMD/NVIDIA waves (32 or 64), so no
/// lane sits idle. The right number is hardware-specific and the difference is
/// real, but tuning it per adapter is a v0.7 problem — PLAN.md §25 is explicit
/// that GPU behaviour divergence is a *testing* problem needing real hardware
/// on each platform, which Fusion does not have yet.
pub const WORKGROUP: u32 = 64;

/// The uniform block every kernel binds, at the binding after its buffers.
///
/// One fixed shape rather than a per-kernel struct, because the alternative is
/// a layout description per kernel and a way to express it, and this is an
/// alpha. `count` is the element count every kernel needs to bounds-check
/// itself against; the rest is room for whatever else a kernel wants — a
/// timestep, a smoothing radius, a cell size.
///
/// Sized and aligned to 32 bytes, and the WGSL declaration that matches it is
/// this one — not the one anybody would write first:
///
/// ```wgsl
/// struct Params {
///     count: u32,
///     u0: u32, u1: u32, u2: u32,
///     f: vec4<f32>,
/// };
/// ```
///
/// Writing `u: vec3<u32>` for the three integer knobs is the mistake to know
/// about. A `vec3` in WGSL is aligned to 16 bytes, so it would sit at offset
/// 16 and push `f` to 32, while the Rust `[u32; 3]` sits at offset 4. Nothing
/// warns: the kernel reads the wrong words and produces plausible numbers.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Params {
    /// Number of elements the dispatch covers. Every kernel returns early past
    /// this, because a dispatch is rounded up to whole workgroups.
    pub count: u32,
    /// Three integer knobs.
    pub u: [u32; 3],
    /// Four float knobs.
    pub f: [f32; 4],
}

impl Params {
    pub fn count(count: usize) -> Self {
        Self {
            count: count as u32,
            ..Default::default()
        }
    }

    pub fn with_f(mut self, f: [f32; 4]) -> Self {
        self.f = f;
        self
    }

    pub fn with_u(mut self, u: [u32; 3]) -> Self {
        self.u = u;
        self
    }
}

/// A compiled compute kernel over `buffers` storage buffers.
///
/// The binding convention is fixed and is the whole of the interface: storage
/// buffers at `@group(0) @binding(0 .. buffers)`, then [`Params`] as a uniform
/// at `@binding(buffers)`. A convention rather than reflection because naga
/// can tell us the layout and matching it up with what the caller meant is
/// more machinery than a rule that fits in a sentence.
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: usize,
    workgroup: u32,
    strided: bool,
    label: String,
}

impl Kernel {
    /// Compile `wgsl` and take `entry` from it.
    ///
    /// Shader compilation failures panic rather than returning an error, and
    /// deliberately: a kernel's WGSL is a source file in this repository, not
    /// user input, so a compile failure is a build error that happens to have
    /// been deferred to run time. wgpu's error carries the line and column.
    pub fn new(device: &Device, label: &str, wgsl: &str, entry: &str, buffers: usize) -> Self {
        Self::sized(device, label, wgsl, entry, buffers, WORKGROUP)
    }

    /// The same, for a kernel whose `@workgroup_size` is not [`WORKGROUP`].
    ///
    /// It has to be told rather than discovered, because the number here and
    /// the number in the WGSL are two statements of one fact and only the
    /// shader's is binding. A scan wants 256 and a per-bucket sort wants
    /// whatever keeps a workgroup busy; a neighbour sweep wants 64.
    pub fn sized(
        device: &Device,
        label: &str,
        wgsl: &str,
        entry: &str,
        buffers: usize,
        workgroup: u32,
    ) -> Self {
        let module = device
            .raw()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..buffers)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: buffers as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                // Dynamic, so one buffer can hold every dispatch's parameters
                // in a graph and each bind group picks its own slice out of
                // it. See `Graph::run`.
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });

        let layout = device
            .raw()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        let pipeline_layout =
            device
                .raw()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
        let pipeline = device
            .raw()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            layout,
            buffers,
            workgroup: workgroup.max(1),
            strided: false,
            label: label.to_string(),
        }
    }

    /// Declare that this kernel loops over its elements with a grid stride —
    /// `for (var i = gid.x; i < count; i += num_workgroups.x * WG)` — rather
    /// than assuming one element per thread.
    ///
    /// It has to be declared because of what happens otherwise. WebGPU caps a
    /// dispatch at 65 535 workgroups per dimension, and that is the *maximum*
    /// an adapter may report rather than a conservative default — at 64
    /// threads a workgroup it is a ceiling of 4.2 million elements. A graph
    /// that quietly clamped the dispatch to the limit would leave everything
    /// past it untouched, which is a fluid that is correct in the middle and
    /// stale at the edges. So a kernel that has not said it can handle a short
    /// dispatch gets a panic naming itself instead.
    #[must_use]
    pub fn strided(mut self) -> Self {
        self.strided = true;
        self
    }

    /// Whether this kernel loops over a grid stride. See [`Kernel::strided`].
    pub fn is_strided(&self) -> bool {
        self.strided
    }

    /// Threads per workgroup, as declared in the WGSL.
    pub fn workgroup(&self) -> u32 {
        self.workgroup
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn buffers(&self) -> usize {
        self.buffers
    }

    pub(crate) fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("label", &self.label)
            .field("buffers", &self.buffers)
            .finish()
    }
}
