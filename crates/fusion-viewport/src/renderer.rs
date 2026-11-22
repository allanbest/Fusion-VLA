//! wgpu plumbing for the viewport.
//!
//! PLAN.md §24 picks wgpu so that one developer does not hand-maintain Vulkan,
//! Metal and DX12 backends. §18 sets the ceiling this module aims at — "a good
//! real-time viewport: PBR shading, shadow maps, SSAO, tone mapping" — and,
//! just as importantly, the ceiling it must not exceed. Production rendering
//! is explicitly not a goal: anyone wanting film quality already owns a
//! renderer, and Fusion exports to it.
//!
//! The pass order and the reasoning behind it are documented at the top of
//! `shader.wgsl`, next to the code that implements them.
//!
//! ## Targets
//!
//! Lighting happens in `Rgba16Float` and is tone-mapped down at the end. A
//! renderer that lights directly into an 8-bit swapchain has already clipped
//! every highlight before it can roll them off, and no amount of grading
//! afterwards brings them back.

use std::sync::Arc;

use crate::camera::{self, Mat4, OrbitCamera};
use crate::mesh::{Geometry, Vertex};

/// Sun direction — the vector *towards* the light.
const LIGHT_DIRECTION: [f32; 3] = [0.42, 0.78, 0.46];
const SHADOW_RESOLUTION: u32 = 2048;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_projection: Mat4,
    view: Mat4,
    projection: Mat4,
    inverse_projection: Mat4,
    inverse_view: Mat4,
    light_view_projection: Mat4,
    eye: [f32; 4],
    light: [f32; 4],
    render: [f32; 4],
    grade: [f32; 4],
}

/// The dials §18's "good viewport" leaves to taste.
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    /// Linear exposure multiplier before tone mapping.
    pub exposure: f32,
    /// Strength of the hemispherical ambient term.
    pub ambient: f32,
    /// Shadow map filter width, in texels.
    pub shadow_softness: f32,
    /// SSAO sampling radius, in world units.
    pub ao_radius: f32,
    /// How much of the ambient AO is allowed to remove. Zero disables it.
    pub ao_strength: f32,
    /// Half-width of the shadow map's orthographic box, in world units.
    ///
    /// A directional light has no natural extent, so something has to choose
    /// one. Too small and distant geometry stops casting; too large and the
    /// same 2048 texels are spread over a bigger area, so contact shadows
    /// coarsen. The viewport fits this to the scene each frame.
    pub shadow_extent: f32,
    /// A plain studio backdrop in place of the sky: a grey gradient, for
    /// looking at one thing on its own — the Materials bench — rather than a
    /// scene in a world.
    pub studio: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            ambient: 0.55,
            shadow_softness: 1.0,
            ao_radius: 0.45,
            ao_strength: 0.9,
            shadow_extent: 24.0,
            studio: false,
        }
    }
}

/// Where the frame is going: a window, or memory.
enum Target {
    Surface {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    /// Offscreen, for screenshots and for tests that want to look at a frame
    /// without a display (PLAN.md §25 — the research workflow has no screen).
    Offscreen {
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    },
}

/// One frame, opened up for a caller to draw into. See
/// [`Renderer::render_with`].
pub struct Overlay<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// The frame's encoder. Anything recorded here lands in the same
    /// submission as the scene.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The tone-mapped, presentable image. Load it rather than clearing it.
    pub target: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: Target,
    /// Pixel format of the presentable image. Kept so an overlay can build a
    /// pipeline that matches it; see [`Renderer::render_with`].
    output_format: wgpu::TextureFormat,

    depth_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    ao_view: wgpu::TextureView,
    ao_blur_view: wgpu::TextureView,
    shadow_view: wgpu::TextureView,

    depth_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    triangle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,
    volume_pipeline: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    frame_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    source_layout: wgpu::BindGroupLayout,
    volume_layout: wgpu::BindGroupLayout,
    frame_bind_group: wgpu::BindGroup,
    lighting_bind_group: wgpu::BindGroup,
    ssao_source: wgpu::BindGroup,
    blur_source: wgpu::BindGroup,
    tonemap_source: wgpu::BindGroup,
    volume_bind_group: wgpu::BindGroup,

    /// The lattice on the GPU, and the one it was filled from.
    ///
    /// The `Arc` is kept rather than only its address: comparing pointers to
    /// decide whether to skip an upload is only sound while the old allocation
    /// cannot be freed and a new one land at the same address.
    volume_uniform: wgpu::Buffer,
    volume_texture: wgpu::Texture,
    volume_view: wgpu::TextureView,
    volume_lut: wgpu::Texture,
    volume_lut_view: wgpu::TextureView,
    volume_current: Option<std::sync::Arc<crate::Volume>>,

    triangle_buffer: wgpu::Buffer,
    triangle_capacity: usize,
    line_buffer: wgpu::Buffer,
    line_capacity: usize,
    overlay_buffer: wgpu::Buffer,
    overlay_capacity: usize,

    pub settings: RenderSettings,
    pub clear_color: wgpu::Color,
}

impl Renderer {
    pub fn new(window: Arc<winit::window::Window>) -> Self {
        pollster::block_on(Self::new_windowed(window))
    }

    /// A renderer with no window, drawing into a texture it owns.
    pub fn offscreen(width: u32, height: u32) -> Self {
        Self::try_offscreen(width, height).expect("no GPU adapter available")
    }

    /// As [`Self::offscreen`], but `None` when the machine has no usable GPU.
    ///
    /// The render tests use this. A build box with no adapter should skip them
    /// and say so, not fail — the alternative is either a red CI on hardware
    /// that was never going to run a viewport, or no render tests at all.
    pub fn try_offscreen(width: u32, height: u32) -> Option<Self> {
        pollster::block_on(Self::new_offscreen(width, height))
    }

    async fn new_windowed(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).expect("create surface");
        let (adapter, device, queue) = request_device(&instance, Some(&surface)).await;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let max_dimension = device.limits().max_texture_dimension_2d;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.clamp(1, max_dimension),
            height: size.height.clamp(1, max_dimension),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let (width, height) = (config.width, config.height);
        Self::assemble(
            device,
            queue,
            Target::Surface { surface, config },
            format,
            width,
            height,
        )
    }

    async fn new_offscreen(width: u32, height: u32) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let (_adapter, device, queue) = try_request_device(&instance, None).await?;
        // Not sRGB: a screenshot is read back and written to a file, and the
        // file format carries its own colour space. Rendering into an sRGB
        // target and then writing the bytes verbatim encodes the transfer
        // twice.
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Some(Self::assemble(
            device,
            queue,
            Target::Offscreen {
                texture,
                width,
                height,
            },
            format,
            width,
            height,
        ))
    }

    fn assemble(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: Target,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame layout"),
            entries: &[uniform_entry(0), filtering_sampler_entry(1)],
        });
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting layout"),
            entries: &[
                depth_texture_entry(0),
                comparison_sampler_entry(1),
                texture_entry(2),
            ],
        });
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("source layout"),
            entries: &[texture_entry(3), depth_texture_entry(4)],
        });
        // The volume pass reads the scene's depth to know where to stop, its
        // own placement, the lattice, and the colour ramp. Group 1 again, with
        // different contents — see the note above about why that is legal and
        // why it has to be a separate layout.
        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume layout"),
            entries: &[
                depth_texture_entry(4),
                uniform_entry(5),
                texture3d_entry(6),
                texture_entry(7),
            ],
        });

        // Two layouts, and the split is not cosmetic. A geometry pass writes
        // the depth buffer; a fullscreen pass reads it. Giving the geometry
        // pipelines a layout that includes the source group means binding the
        // depth texture in the same pass that is writing it, which is a
        // read-write hazard and a validation error — the depth prepass failed
        // with "encoder is invalid" until the layouts were separated.
        let geometry_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth-only layout"),
            bind_group_layouts: &[&frame_layout],
            push_constant_ranges: &[],
        });
        let shaded_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaded layout"),
            bind_group_layouts: &[&frame_layout, &lighting_layout],
            push_constant_ranges: &[],
        });
        let fullscreen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fullscreen layout"),
            bind_group_layouts: &[&frame_layout, &source_layout],
            push_constant_ranges: &[],
        });
        let volume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("volume layout"),
                bind_group_layouts: &[&frame_layout, &volume_layout],
                push_constant_ranges: &[],
            });

        let geometry_pipeline = |entry: &str,
                                 fragment: Option<(&str, wgpu::TextureFormat)>,
                                 topology,
                                 cull,
                                 depth_write,
                                 depth_compare,
                                 label: &str| {
            let targets = fragment.map(|(_, format)| {
                [Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(if fragment.is_some() {
                    &shaded_layout
                } else {
                    &geometry_layout
                }),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: fragment.map(|(name, _)| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(name),
                    targets: targets.as_ref().unwrap(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: depth_write,
                    depth_compare,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };

        let depth_pipeline = geometry_pipeline(
            "vs_depth",
            None,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            true,
            wgpu::CompareFunction::Less,
            "depth prepass",
        );
        // Front faces culled when casting. A shadow caster's back faces are
        // the ones furthest from the light, so rendering those puts the
        // recorded depth behind the surface that will be tested against it —
        // which is a depth bias that costs nothing and scales itself.
        let shadow_pipeline = geometry_pipeline(
            "vs_shadow",
            None,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Front),
            true,
            wgpu::CompareFunction::Less,
            "shadow",
        );
        // The main pass reuses the prepass's depth: equal, and no writes.
        let triangle_pipeline = geometry_pipeline(
            "vs_main",
            Some(("fs_main", HDR_FORMAT)),
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            false,
            wgpu::CompareFunction::Equal,
            "triangles",
        );
        // Lines are never culled and are not in the prepass, so they test
        // against depth normally and write it. An overlay visible from only
        // one side is worse than useless in a debug view.
        let line_pipeline = geometry_pipeline(
            "vs_main",
            Some(("fs_main", HDR_FORMAT)),
            wgpu::PrimitiveTopology::LineList,
            None,
            true,
            wgpu::CompareFunction::LessEqual,
            "lines",
        );

        let fullscreen = |entry: &str, format, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&fullscreen_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        // Overlay: no depth at all, and drawn into the tone-mapped output
        // rather than the HDR target. A legend is a UI element, not part of
        // the scene — tone mapping a colour bar would make the swatch at the
        // top of the ramp a different colour from the one the data uses.
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&shaded_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_overlay"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        // The volume marcher: a fullscreen triangle, composited over the
        // tone-mapped image with premultiplied alpha. Premultiplied because
        // the march accumulates `colour += (1 - alpha) * a * rgb` front to
        // back, which *is* the premultiplied form — blending it as straight
        // alpha would multiply the coverage in twice and every field would
        // render dark.
        let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume"),
            layout: Some(&volume_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_volume"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let ssao_pipeline = fullscreen("fs_ssao", AO_FORMAT, "ssao");
        let blur_pipeline = fullscreen("fs_blur", AO_FORMAT, "ao blur");
        let tonemap_pipeline = fullscreen("fs_tonemap", output_format, "tonemap");

        let shadow_view = create_texture(
            &device,
            SHADOW_RESOLUTION,
            SHADOW_RESOLUTION,
            DEPTH_FORMAT,
            "shadow map",
        );
        let (depth_view, hdr_view, ao_view, ao_blur_view) =
            create_screen_targets(&device, width, height);

        let volume_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume uniforms"),
            size: std::mem::size_of::<VolumeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A single empty voxel until something asks for a volume. A bind group
        // layout has no notion of "nothing bound", so the alternative to a
        // placeholder is a second pipeline layout and a second pipeline for
        // the case where there is no field to draw.
        let (volume_texture, volume_view) = create_volume_texture(&device, [1, 1, 1]);
        let (volume_lut, volume_lut_view) = create_lookup_texture(&device);

        let (triangle_buffer, triangle_capacity) = create_vertex_buffer(&device, 1 << 16);
        let (line_buffer, line_capacity) = create_vertex_buffer(&device, 1 << 16);
        let (overlay_buffer, overlay_capacity) = create_vertex_buffer(&device, 1 << 14);

        let mut renderer = Self {
            output_format,
            frame_bind_group: make_frame_bind_group(&device, &frame_layout, &uniform_buffer),
            lighting_bind_group: make_lighting_bind_group(
                &device,
                &lighting_layout,
                &shadow_view,
                &ao_blur_view,
            ),
            ssao_source: make_source_bind_group(&device, &source_layout, &hdr_view, &depth_view),
            blur_source: make_source_bind_group(&device, &source_layout, &ao_view, &depth_view),
            tonemap_source: make_source_bind_group(&device, &source_layout, &hdr_view, &depth_view),
            volume_bind_group: make_volume_bind_group(
                &device,
                &volume_layout,
                &depth_view,
                &volume_uniform,
                &volume_view,
                &volume_lut_view,
            ),
            device,
            queue,
            target,
            depth_view,
            hdr_view,
            ao_view,
            ao_blur_view,
            shadow_view,
            depth_pipeline,
            shadow_pipeline,
            triangle_pipeline,
            line_pipeline,
            overlay_pipeline,
            ssao_pipeline,
            blur_pipeline,
            tonemap_pipeline,
            volume_pipeline,
            uniform_buffer,
            frame_layout,
            lighting_layout,
            source_layout,
            volume_layout,
            volume_uniform,
            volume_texture,
            volume_view,
            volume_lut,
            volume_lut_view,
            volume_current: None,
            triangle_buffer,
            triangle_capacity,
            line_buffer,
            line_capacity,
            overlay_buffer,
            overlay_capacity,
            settings: RenderSettings::default(),
            clear_color: wgpu::Color {
                r: 0.021,
                g: 0.024,
                b: 0.031,
                a: 1.0,
            },
        };
        renderer.rebuild_bind_groups();
        renderer
    }

    fn rebuild_bind_groups(&mut self) {
        self.frame_bind_group =
            make_frame_bind_group(&self.device, &self.frame_layout, &self.uniform_buffer);
        self.lighting_bind_group = make_lighting_bind_group(
            &self.device,
            &self.lighting_layout,
            &self.shadow_view,
            &self.ao_blur_view,
        );
        // SSAO reads depth; the blur reads the raw AO; tonemap reads the HDR
        // target. Each needs both slots filled, so the unused one is given
        // whatever is to hand — a bind group layout has no notion of
        // "optional".
        //
        // The colour slot of the SSAO group is the HDR target and *not* the AO
        // texture, even though the AO texture would be the intuitive filler:
        // the SSAO pass renders *into* the AO texture, and binding a texture
        // that a pass is also writing is a read-write hazard. wgpu reports it
        // as "encoder is invalid" on the *next* pass, which is a long way from
        // where the mistake is.
        self.ssao_source = make_source_bind_group(
            &self.device,
            &self.source_layout,
            &self.hdr_view,
            &self.depth_view,
        );
        self.blur_source = make_source_bind_group(
            &self.device,
            &self.source_layout,
            &self.ao_view,
            &self.depth_view,
        );
        self.tonemap_source = make_source_bind_group(
            &self.device,
            &self.source_layout,
            &self.hdr_view,
            &self.depth_view,
        );
        self.volume_bind_group = make_volume_bind_group(
            &self.device,
            &self.volume_layout,
            &self.depth_view,
            &self.volume_uniform,
            &self.volume_view,
            &self.volume_lut_view,
        );
    }

    /// Hand the renderer a scalar field to look inside, or `None` for none.
    ///
    /// Uploads only when the lattice is a different one — the editor rebuilds
    /// a volume when the simulation steps and hands the same `Arc` over on
    /// every frame in between, and a 128³ lattice is 16 MB that nobody should
    /// be pushing across the bus sixty times a second to say "unchanged".
    pub fn set_volume(&mut self, volume: Option<&std::sync::Arc<crate::Volume>>) {
        let Some(volume) = volume else {
            self.volume_current = None;
            self.queue.write_buffer(
                &self.volume_uniform,
                0,
                bytemuck::bytes_of(&VolumeUniforms::absent()),
            );
            return;
        };
        let dims = [
            volume.dims[0] as u32,
            volume.dims[1] as u32,
            volume.dims[2] as u32,
        ];
        let same = self
            .volume_current
            .as_ref()
            .is_some_and(|held| std::sync::Arc::ptr_eq(held, volume));
        if !same {
            let size = self.volume_texture.size();
            if [size.width, size.height, size.depth_or_array_layers] != dims {
                let (texture, view) = create_volume_texture(&self.device, dims);
                self.volume_texture = texture;
                self.volume_view = view;
                self.volume_bind_group = make_volume_bind_group(
                    &self.device,
                    &self.volume_layout,
                    &self.depth_view,
                    &self.volume_uniform,
                    &self.volume_view,
                    &self.volume_lut_view,
                );
            }
            let texels = volume.texels();
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.volume_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims[0] * 8),
                    rows_per_image: Some(dims[1]),
                },
                wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                },
            );
            let lookup = volume.lookup();
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.volume_lut,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&lookup),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.volume_current = Some(volume.clone());
        }
        self.queue.write_buffer(
            &self.volume_uniform,
            0,
            bytemuck::bytes_of(&VolumeUniforms::of(volume)),
        );
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        // Clamped rather than trusted: a surface larger than the device's
        // maximum texture is a validation error raised on a non-unwinding
        // callback, which aborts the process instead of returning an error.
        let max_dimension = self.device.limits().max_texture_dimension_2d;
        let (width, height) = (width.min(max_dimension), height.min(max_dimension));
        match &mut self.target {
            Target::Surface { surface, config } => {
                config.width = width;
                config.height = height;
                surface.configure(&self.device, config);
            }
            Target::Offscreen { .. } => return,
        }
        let (depth, hdr, ao, ao_blur) = create_screen_targets(&self.device, width, height);
        self.depth_view = depth;
        self.hdr_view = hdr;
        self.ao_view = ao;
        self.ao_blur_view = ao_blur;
        self.rebuild_bind_groups();
    }

    /// The GPU device this renderer owns.
    ///
    /// Exposed for the same reason [`Renderer::render_with`] is: an overlay
    /// that draws through wgpu has to build its pipelines against the device
    /// that will run them, and it has to do that before the first frame rather
    /// than during one.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Pixel format of the presentable image.
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    pub fn size(&self) -> (u32, u32) {
        match &self.target {
            Target::Surface { config, .. } => (config.width, config.height),
            Target::Offscreen { width, height, .. } => (*width, *height),
        }
    }

    pub fn aspect(&self) -> f32 {
        let (w, h) = self.size();
        w as f32 / h.max(1) as f32
    }

    /// Fit the shadow map's box to what is on screen.
    ///
    /// Without this the extent is a guess, and a guess is wrong in both
    /// directions: a pit demo two metres across gets a 24 m box and shadows
    /// quantised to a centimetre, while a scene fifty metres across has its
    /// edges silently stop casting.
    fn fit_shadows(&self, geometry: &Geometry, camera: &OrbitCamera) -> (Mat4, f32) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &geometry.triangles {
            for k in 0..3 {
                min[k] = min[k].min(v.position[k]);
                max[k] = max[k].max(v.position[k]);
            }
        }
        if !min[0].is_finite() {
            min = [-1.0; 3];
            max = [1.0; 3];
        }
        let centre = fusion_math::vec3(
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        );
        let radius = {
            let d = fusion_math::vec3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
            (d.length() * 0.5).max(1.0)
        };
        // A scene can be enormous — a 40 m floor plane under a 2 m stack — and
        // spending the whole shadow map on the floor makes the stack's
        // shadows useless. Cap the box at a multiple of the camera's distance,
        // which is a decent proxy for what the viewer is actually looking at.
        let extent = radius.min(camera.distance * 1.6).max(2.0);
        let focus = camera.focus;
        let light = fusion_math::Vec3::from_array(LIGHT_DIRECTION).normalize_or_zero();
        // Far enough back that nothing in the scene is behind the near plane.
        let eye = focus + light * (radius + extent);
        let view = camera::look_at(eye, focus, fusion_math::Vec3::Y);
        let projection =
            camera::orthographic(extent, extent, 0.05, radius * 2.0 + extent * 2.0 + 1.0);
        let _ = centre;
        (camera::multiply(&projection, &view), extent)
    }

    pub fn render(&mut self, camera: &OrbitCamera, geometry: &Geometry) {
        self.render_with(camera, geometry, |_| {});
    }

    /// Render one frame, letting the caller record into the same command
    /// encoder against the finished image before it is submitted.
    ///
    /// This is the seam the sandbox GUI hangs off. PLAN.md §25 abstracts GUI
    /// widgets behind egui, and egui draws through wgpu — so it needs the
    /// device, the encoder and the target view that this renderer owns. The
    /// alternative is a second surface and a second submission a frame, which
    /// is both slower and a source of tearing between the two halves of one
    /// picture.
    ///
    /// The viewport still knows nothing about GUIs. It hands out a frame and
    /// whoever asked for it draws what they like.
    pub fn render_with(
        &mut self,
        camera: &OrbitCamera,
        geometry: &Geometry,
        overlay: impl FnOnce(Overlay<'_>),
    ) {
        let (width, height) = self.size();
        let aspect = self.aspect();
        let eye = camera.eye();
        let view = camera.view();
        let projection = camera.projection(aspect);
        let (light_view_projection, extent) = self.fit_shadows(geometry, camera);
        let _ = extent;

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_projection: camera::multiply(&projection, &view),
                view,
                projection,
                inverse_projection: camera::inverse(&projection),
                inverse_view: camera::inverse(&view),
                light_view_projection,
                eye: [eye.x, eye.y, eye.z, 1.0],
                light: [
                    LIGHT_DIRECTION[0],
                    LIGHT_DIRECTION[1],
                    LIGHT_DIRECTION[2],
                    1.0 / SHADOW_RESOLUTION as f32,
                ],
                render: [
                    width as f32,
                    height as f32,
                    self.settings.ao_radius,
                    self.settings.ao_strength,
                ],
                grade: [
                    self.settings.exposure,
                    self.settings.ambient,
                    self.settings.shadow_softness,
                    if self.settings.studio { 1.0 } else { 0.0 },
                ],
            }),
        );

        self.set_volume(geometry.volume.as_ref());

        upload(
            &self.device,
            &self.queue,
            &geometry.triangles,
            &mut self.triangle_buffer,
            &mut self.triangle_capacity,
        );
        upload(
            &self.device,
            &self.queue,
            &geometry.lines,
            &mut self.line_buffer,
            &mut self.line_capacity,
        );
        upload(
            &self.device,
            &self.queue,
            &geometry.overlay,
            &mut self.overlay_buffer,
            &mut self.overlay_capacity,
        );

        let frame = match &self.target {
            Target::Surface { surface, config } => match surface.get_current_texture() {
                Ok(frame) => Some(frame),
                // Lost or outdated surfaces happen on resize and on display
                // changes; reconfiguring and skipping the frame is the fix.
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    surface.configure(&self.device, config);
                    return;
                }
                Err(wgpu::SurfaceError::Timeout) => return,
                Err(e) => {
                    eprintln!("surface error: {e:?}");
                    return;
                }
            },
            Target::Offscreen { .. } => None,
        };
        let output_view = match (&frame, &self.target) {
            (Some(frame), _) => frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            (None, Target::Offscreen { texture, .. }) => {
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
            _ => unreachable!("a surface target always produces a frame"),
        };

        let triangles = geometry.triangles.len() as u32;
        let lines = geometry.lines.len() as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // --- 1. depth prepass ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(depth_attachment(&self.depth_view, true)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if triangles > 0 {
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                pass.set_pipeline(&self.depth_pipeline);
                pass.set_vertex_buffer(0, self.triangle_buffer.slice(..));
                pass.draw(0..triangles, 0..1);
            }
        }

        // --- 2. SSAO, and 3. blur ---
        self.fullscreen_pass(
            &mut encoder,
            "ssao",
            &self.ssao_pipeline,
            &self.ssao_source,
            &self.ao_view,
        );
        self.fullscreen_pass(
            &mut encoder,
            "ao blur",
            &self.blur_pipeline,
            &self.blur_source,
            &self.ao_blur_view,
        );

        // --- 4. shadow map ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(depth_attachment(&self.shadow_view, true)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if triangles > 0 {
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                pass.set_pipeline(&self.shadow_pipeline);
                pass.set_vertex_buffer(0, self.triangle_buffer.slice(..));
                pass.draw(0..triangles, 0..1);
            }
        }

        // --- 5. main pass ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(depth_attachment(&self.depth_view, false)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(1, &self.lighting_bind_group, &[]);
            if triangles > 0 {
                pass.set_pipeline(&self.triangle_pipeline);
                pass.set_vertex_buffer(0, self.triangle_buffer.slice(..));
                pass.draw(0..triangles, 0..1);
            }
            if lines > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, self.line_buffer.slice(..));
                pass.draw(0..lines, 0..1);
            }
        }

        // --- 6. tone map ---
        self.fullscreen_pass(
            &mut encoder,
            "tonemap",
            &self.tonemap_pipeline,
            &self.tonemap_source,
            &output_view,
        );

        // --- 6.5. the volume, over the tone-mapped image ---
        //
        // After tone mapping rather than before it, because the colours come
        // from a fixed scale (§19) and are already the colours the legend
        // shows. Running them through ACES would make a temperature read as a
        // different colour depending on how bright the rest of the scene was,
        // which is the whole failure a fixed scale exists to prevent.
        if self.volume_current.is_some() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(1, &self.volume_bind_group, &[]);
            pass.set_pipeline(&self.volume_pipeline);
            pass.draw(0..3, 0..1);
        }

        // --- 7. overlay ---
        if !geometry.overlay.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(1, &self.lighting_bind_group, &[]);
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_vertex_buffer(0, self.overlay_buffer.slice(..));
            pass.draw(0..geometry.overlay.len() as u32, 0..1);
        }

        // --- 8. whatever the caller wants on top ---
        overlay(Overlay {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut encoder,
            target: &output_view,
            format: self.output_format,
            width,
            height,
        });

        self.queue.submit(Some(encoder.finish()));
        if let Some(frame) = frame {
            frame.present();
        }
    }

    fn fullscreen_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::BindGroup,
        target: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, &self.frame_bind_group, &[]);
        pass.set_bind_group(1, source, &[]);
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }

    /// Render one frame and read it back as RGBA8 rows.
    ///
    /// Offscreen renderers only. Used by `screenshot` and by the render tests,
    /// which is the point: a renderer nothing can look at without a display is
    /// a renderer nothing can test.
    pub fn read_back(&mut self, camera: &OrbitCamera, geometry: &Geometry) -> Option<Image> {
        self.read_back_with(camera, geometry, |_| {})
    }

    /// The same, with an overlay — so whatever draws on top of a frame can be
    /// tested on a machine with no window. See [`Renderer::render_with`].
    pub fn read_back_with(
        &mut self,
        camera: &OrbitCamera,
        geometry: &Geometry,
        overlay: impl FnOnce(Overlay<'_>),
    ) -> Option<Image> {
        let Target::Offscreen {
            texture,
            width,
            height,
            ..
        } = &self.target
        else {
            return None;
        };
        let (width, height) = (*width, *height);
        let texture = texture.clone();
        self.render_with(camera, geometry, overlay);

        // wgpu requires each row of a buffer copy to start on a 256-byte
        // boundary, so the readback buffer is padded and unpadded here rather
        // than the caller having to know.
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height {
            let start = (row * padded) as usize;
            pixels.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        buffer.unmap();
        Some(Image {
            width,
            height,
            pixels,
        })
    }
}

/// An RGBA8 image read back from the GPU.
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Row-major, four bytes per pixel.
    pub pixels: Vec<u8>,
}

impl Image {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Mean luminance, 0..1. A cheap way for a test to say "something was
    /// drawn" and "it is not all one colour".
    pub fn mean_luminance(&self) -> f32 {
        let mut sum = 0.0f64;
        for p in self.pixels.chunks_exact(4) {
            sum += (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64) / 255.0;
        }
        (sum / (self.width * self.height) as f64) as f32
    }
}

async fn request_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    try_request_device(instance, surface)
        .await
        .expect("no suitable GPU adapter")
}

async fn try_request_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
) -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("fusion-viewport"),
            required_features: wgpu::Features::empty(),
            // Conservative baseline, but with the resolution limits raised to
            // whatever this adapter actually supports.
            //
            // `downlevel_defaults` caps a texture at 2048 on a side, which
            // sounds generous until a window is opened near full screen on a
            // Retina display: the *physical* surface is then 3024 x 1898 and
            // configuring it fails with a validation error, inside a callback
            // that cannot unwind, so the process aborts rather than reporting
            // anything useful.
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .ok()?;
    Some((adapter, device, queue))
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn comparison_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    }
}

fn filtering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn make_frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let linear = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&linear),
            },
        ],
    })
}

fn make_lighting_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shadow: &wgpu::TextureView,
    ao: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lighting bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(shadow),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(ao),
            },
        ],
    })
}

fn make_source_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("source bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(color),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(depth),
            },
        ],
    })
}

fn depth_attachment(
    view: &wgpu::TextureView,
    clear: bool,
) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: if clear {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_screen_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
) {
    (
        create_texture(device, width, height, DEPTH_FORMAT, "depth"),
        create_texture(device, width, height, HDR_FORMAT, "hdr"),
        create_texture(device, width, height, AO_FORMAT, "ao"),
        create_texture(device, width, height, AO_FORMAT, "ao blur"),
    )
}

fn create_vertex_buffer(device: &wgpu::Device, vertices: usize) -> (wgpu::Buffer, usize) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertices"),
        size: (vertices * std::mem::size_of::<Vertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (buffer, vertices)
}

/// Write vertices, growing the buffer geometrically so a scene that gains
/// detail does not reallocate every frame.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[Vertex],
    buffer: &mut wgpu::Buffer,
    capacity: &mut usize,
) {
    if data.is_empty() {
        return;
    }
    if data.len() > *capacity {
        let (new_buffer, cap) = create_vertex_buffer(device, data.len().next_power_of_two());
        *buffer = new_buffer;
        *capacity = cap;
    }
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}

/// Placement and marching parameters for the volume pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumeUniforms {
    /// xyz: world position of voxel (0, 0, 0). w: voxel edge length.
    origin: [f32; 4],
    /// xyz: voxel counts. w: opacity of full material per metre.
    dims: [f32; 4],
    /// x: step ceiling. y: step length in voxels. z: 1 where a volume is
    /// bound. w: unused.
    march: [f32; 4],
}

impl VolumeUniforms {
    /// The "nothing to draw" state. The pass is skipped on the CPU as well;
    /// this is the belt to that pair of braces, because a stale uniform with a
    /// live flag would march a texture that had been resized underneath it.
    fn absent() -> Self {
        Self {
            origin: [0.0; 4],
            dims: [1.0, 1.0, 1.0, 0.0],
            march: [0.0, 0.5, 0.0, 0.0],
        }
    }

    fn of(volume: &crate::Volume) -> Self {
        // Half a voxel per step: the Nyquist rate for a lattice, and the point
        // past which a finer step buys nothing because the trilinear filter
        // has no more detail to give. The ceiling is what stops a camera
        // inside a large volume costing a thousand samples a pixel.
        Self {
            origin: [
                volume.origin.x,
                volume.origin.y,
                volume.origin.z,
                volume.cell,
            ],
            dims: [
                volume.dims[0] as f32,
                volume.dims[1] as f32,
                volume.dims[2] as f32,
                volume.density,
            ],
            march: [512.0, 0.5, 1.0, volume.base],
        }
    }
}

/// The lattice, as a filterable 3D texture.
///
/// `Rgba16Float` rather than `R32Float`: single-precision textures are not
/// filterable without a device feature, and an unfiltered volume renders as
/// visible cubes. Half precision is four decimal digits on a field already
/// normalized to `[0, 1]`, against a colour ramp with 256 levels in it.
fn create_volume_texture(
    device: &wgpu::Device,
    dims: [u32; 3],
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume"),
        size: wgpu::Extent3d {
            width: dims[0].max(1),
            height: dims[1].max(1),
            depth_or_array_layers: dims[2].max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The colour ramp, as 256 texels of sRGB.
///
/// sRGB-encoded and sampled through an sRGB format, so what the shader reads
/// back is the linear colour the `ScalarField` produced — the same number the
/// legend and the material points are drawn with, which is what makes the
/// volume and the points agree about what 400 K looks like.
fn create_lookup_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume colour ramp"),
        size: wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn make_volume_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    field: &wgpu::TextureView,
    lookup: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(depth),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(field),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(lookup),
            },
        ],
    })
}

fn texture3d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}
