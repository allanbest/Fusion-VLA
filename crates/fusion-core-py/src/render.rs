//! `fusion_vla.core.Renderer`: GPU camera images of Fusion Core worlds.
//!
//! This draws through the Fusion viewport's offscreen wgpu pipeline (shadows,
//! SSAO, tone mapping), the same renderer the editor uses. It runs headless,
//! so no window or display is needed. One `Renderer` owns one GPU device and
//! one render target of a fixed resolution.
//!
//! Geometry is the collision geometry Fusion Core simulates. Spheres and boxes
//! go through the viewport's own shape drawing. Capsules are tessellated here
//! with smooth normals. Half-spaces become a checkered floor tile. FEM solids
//! are drawn from their surface triangles at the current node positions.

use std::collections::HashMap;
use std::f32::consts::PI;

use core_math::{Quat as CQuat, Vec3 as CVec3};
use core_space::Shape as CShape;
use core_world::{Representation, World};
use fusion_core::Shape as VShape;
use fusion_math::{vec3, Isometry, Quat, Vec3};
use fusion_viewport::{debug_draw, Geometry, OrbitCamera, Renderer};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::core::{PyCoreWorld, PyCoreWorldBatch, PyObjectId};

type Vec3Arg = (f64, f64, f64);

fn v(p: CVec3) -> Vec3 {
    vec3(p.x as f32, p.y as f32, p.z as f32)
}

fn q(r: CQuat) -> Quat {
    Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32)
}

fn arg(p: Vec3Arg) -> PyResult<Vec3> {
    if !(p.0.is_finite() && p.1.is_finite() && p.2.is_finite()) {
        return Err(PyValueError::new_err("camera vectors must be finite"));
    }
    Ok(vec3(p.0 as f32, p.1 as f32, p.2 as f32))
}

/// A unit capsule mesh along local +y: hemispheres of radius 1 joined by a
/// cylinder. Points are stored as (unit direction, which end) so one mesh
/// serves every radius and length.
struct CapsuleMesh {
    dirs: Vec<Vec3>,
    ends: Vec<f32>,
    indices: Vec<u32>,
}

impl CapsuleMesh {
    fn new(segments: u32, rings: u32) -> Self {
        let (mut dirs, mut ends, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        // Rings from the top pole to the bottom pole. The top hemisphere
        // uses end +1 and the bottom one end -1. The equator is duplicated
        // once per end, which makes the cylinder band.
        let mut ring_rows: Vec<(f32, f32)> = Vec::new(); // (polar angle, end)
        for r in 0..=rings {
            ring_rows.push((0.5 * PI * r as f32 / rings as f32, 1.0));
        }
        for r in 0..=rings {
            ring_rows.push((0.5 * PI + 0.5 * PI * r as f32 / rings as f32, -1.0));
        }
        let cols = segments + 1;
        for &(theta, end) in &ring_rows {
            for s in 0..cols {
                let phi = 2.0 * PI * s as f32 / segments as f32;
                dirs.push(vec3(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()));
                ends.push(end);
            }
        }
        for row in 0..(ring_rows.len() as u32 - 1) {
            for s in 0..segments {
                let a = row * cols + s;
                let b = a + cols;
                indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        Self { dirs, ends, indices }
    }
}

struct Frame {
    /// Scene frame to renderer frame (the viewport is y-up).
    to_render: Quat,
}

impl Frame {
    fn point(&self, p: Vec3) -> Vec3 {
        self.to_render * p
    }
    fn pose(&self, p: Vec3, r: Quat) -> Isometry {
        Isometry::new(self.to_render * p, self.to_render * r)
    }
}

#[pyclass(module = "fusion_vla.core", name = "Renderer", unsendable)]
pub struct PyRenderer {
    renderer: Renderer,
    width: u32,
    height: u32,
    frame: Frame,
    up_axis: char,
    capsule: CapsuleMesh,
    colors: HashMap<u32, [f32; 3]>,
    floor_colors: ([f32; 3], [f32; 3]),
    floor_extent: f32,
    floor_tile: f32,
}

impl PyRenderer {
    fn color_for(&self, index: u32, dynamic: bool) -> [f32; 3] {
        self.colors
            .get(&index)
            .copied()
            .unwrap_or_else(|| debug_draw::body_color(index as usize, dynamic))
    }

    fn draw_world(&self, world: &World, g: &mut Geometry) {
        for (i, object) in world.objects().iter().enumerate() {
            let i = i as u32;
            match &object.representation {
                Representation::Fixed(f) => {
                    self.draw_shape(&f.shape, f.pose.position, f.pose.rotation, self.color_for(i, false), g)
                }
                Representation::Rigid(r) => {
                    self.draw_shape(&r.shape, r.pose.position, r.pose.rotation, self.color_for(i, true), g)
                }
                Representation::Solid(s) => {
                    let pos: Vec<Vec3> =
                        s.nodes.position.iter().map(|&p| self.frame.point(v(p))).collect();
                    let color = self.color_for(i, true);
                    for f in &s.surface.faces {
                        let [a, b, c] = f.map(|k| pos[k as usize]);
                        g.triangle_shaded(a, b, c, color);
                    }
                }
            }
        }
    }

    fn draw_shape(&self, shape: &CShape, p: CVec3, r: CQuat, color: [f32; 3], g: &mut Geometry) {
        let pose = self.frame.pose(v(p), q(r));
        match *shape {
            CShape::Sphere { radius } => {
                g.solid_shape(&VShape::Sphere { radius: radius as f32 }, &pose, color)
            }
            CShape::Cuboid { half } => {
                g.solid_shape(&VShape::Cuboid { half_extents: v(half) }, &pose, color)
            }
            CShape::Capsule { radius, half_length } => {
                let (rad, hl) = (radius as f32, half_length as f32);
                let m = &self.capsule;
                let positions: Vec<Vec3> = m
                    .dirs
                    .iter()
                    .zip(&m.ends)
                    .map(|(&d, &e)| pose.transform_point(d * rad + vec3(0.0, hl * e, 0.0)))
                    .collect();
                let normals: Vec<Vec3> =
                    m.dirs.iter().map(|&d| pose.transform_direction(d)).collect();
                g.indexed_surface(&positions, &normals, &m.indices, color);
            }
            CShape::HalfSpace { normal, offset } => {
                // A finite checkered tile on the plane, centred under the origin.
                let n = (self.frame.to_render * v(normal)).normalize();
                let centre = n * offset as f32;
                let helper = if n.y.abs() < 0.9 { vec3(0.0, 1.0, 0.0) } else { vec3(1.0, 0.0, 0.0) };
                let t = n.cross(helper).normalize();
                let b = n.cross(t);
                let tiles = (2.0 * self.floor_extent / self.floor_tile).ceil().max(1.0) as i32;
                let half = tiles as f32 * self.floor_tile * 0.5;
                for ix in 0..tiles {
                    for iz in 0..tiles {
                        let u0 = -half + ix as f32 * self.floor_tile;
                        let w0 = -half + iz as f32 * self.floor_tile;
                        let (u1, w1) = (u0 + self.floor_tile, w0 + self.floor_tile);
                        let at = |u: f32, w: f32| centre + t * u + b * w;
                        let c = if (ix + iz) % 2 == 0 { self.floor_colors.0 } else { self.floor_colors.1 };
                        // Wound so the face normal (b - a) x (c - a) is +n.
                        g.triangle_shaded(at(u0, w0), at(u1, w1), at(u0, w1), c);
                        g.triangle_shaded(at(u0, w0), at(u1, w0), at(u1, w1), c);
                    }
                }
            }
        }
    }

    fn camera(&self, eye: Vec3Arg, target: Vec3Arg, fov_deg: f64) -> PyResult<OrbitCamera> {
        if !(fov_deg.is_finite() && fov_deg > 1.0 && fov_deg < 179.0) {
            return Err(PyValueError::new_err("fov_deg must be in (1, 179)"));
        }
        let (e, t) = (self.frame.point(arg(eye)?), self.frame.point(arg(target)?));
        if (e - t).length() < 1e-6 {
            return Err(PyValueError::new_err("camera eye and target must differ"));
        }
        let mut cam = OrbitCamera {
            fov_y: (fov_deg as f32).to_radians(),
            near: 0.01,
            far: 500.0,
            ..OrbitCamera::default()
        };
        cam.look_from(e, t);
        Ok(cam)
    }

    fn render_rgb(&mut self, world: &World, cam: &OrbitCamera, out: &mut Vec<u8>) -> PyResult<()> {
        let mut g = Geometry::new();
        self.draw_world(world, &mut g);
        let image = self
            .renderer
            .read_back(cam, &g)
            .ok_or_else(|| PyRuntimeError::new_err("offscreen readback failed"))?;
        out.reserve((self.width * self.height * 3) as usize);
        for px in image.pixels.chunks_exact(4) {
            out.extend_from_slice(&px[..3]);
        }
        Ok(())
    }
}

#[pymethods]
impl PyRenderer {
    /// `Renderer(width, height, up_axis="y")`. `up_axis` is the scene's up
    /// direction: "y" for a bare `World()`, "z" for most MJCF files. Raises
    /// `RuntimeError` when the machine has no usable GPU adapter.
    #[new]
    #[pyo3(signature = (width = 224, height = 224, up_axis = "y"))]
    fn new(width: u32, height: u32, up_axis: &str) -> PyResult<Self> {
        if !(8..=8192).contains(&width) || !(8..=8192).contains(&height) {
            return Err(PyValueError::new_err("width and height must be in 8..=8192"));
        }
        let (to_render, up) = match up_axis {
            "y" | "Y" => (Quat::IDENTITY, 'y'),
            // Scene (x, y, z) -> renderer (x, z, -y).
            "z" | "Z" => (Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), -0.5 * PI), 'z'),
            _ => return Err(PyValueError::new_err("up_axis must be 'y' or 'z'")),
        };
        let mut renderer = Renderer::try_offscreen(width, height).ok_or_else(|| {
            PyRuntimeError::new_err(
                "no usable GPU adapter for offscreen rendering (wgpu found none: \
                 on Linux install Vulkan drivers, e.g. the NVIDIA driver or mesa-vulkan-drivers)",
            )
        })?;
        renderer.settings.shadow_extent = 4.0;
        Ok(Self {
            renderer,
            width,
            height,
            frame: Frame { to_render },
            up_axis: up,
            capsule: CapsuleMesh::new(24, 6),
            colors: HashMap::new(),
            floor_colors: ([0.42, 0.43, 0.45], [0.30, 0.31, 0.33]),
            floor_extent: 4.0,
            floor_tile: 0.25,
        })
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.height
    }
    #[getter]
    fn up_axis(&self) -> String {
        self.up_axis.to_string()
    }

    /// Override one object's colour (linear RGB in 0..1). Applies to every env.
    fn set_color(&mut self, object: PyObjectId, rgb: (f32, f32, f32)) -> PyResult<()> {
        for c in [rgb.0, rgb.1, rgb.2] {
            if !(0.0..=1.0).contains(&c) {
                return Err(PyValueError::new_err("colour components must be in 0..1"));
            }
        }
        self.colors.insert(object.inner.0, [rgb.0, rgb.1, rgb.2]);
        Ok(())
    }

    /// Floor drawing for half-spaces: half-width of the tile in metres, the
    /// checker size, and the two checker colours.
    #[pyo3(signature = (extent = 4.0, tile = 0.25, color_a = (0.42, 0.43, 0.45), color_b = (0.30, 0.31, 0.33)))]
    fn set_floor(
        &mut self,
        extent: f32,
        tile: f32,
        color_a: (f32, f32, f32),
        color_b: (f32, f32, f32),
    ) -> PyResult<()> {
        if !(extent > 0.0 && tile > 0.0 && extent / tile <= 200.0) {
            return Err(PyValueError::new_err("need extent > 0, tile > 0 and extent/tile <= 200"));
        }
        self.floor_extent = extent;
        self.floor_tile = tile;
        self.floor_colors = ([color_a.0, color_a.1, color_a.2], [color_b.0, color_b.1, color_b.2]);
        Ok(())
    }

    /// Lighting knobs from the viewport's `RenderSettings`.
    #[pyo3(signature = (exposure = None, ambient = None, shadow_extent = None, ao_strength = None, studio = None))]
    fn set_lighting(
        &mut self,
        exposure: Option<f32>,
        ambient: Option<f32>,
        shadow_extent: Option<f32>,
        ao_strength: Option<f32>,
        studio: Option<bool>,
    ) {
        let s = &mut self.renderer.settings;
        if let Some(x) = exposure {
            s.exposure = x;
        }
        if let Some(x) = ambient {
            s.ambient = x;
        }
        if let Some(x) = shadow_extent {
            s.shadow_extent = x;
        }
        if let Some(x) = ao_strength {
            s.ao_strength = x;
        }
        if let Some(x) = studio {
            s.studio = x;
        }
    }

    /// One RGB frame of `world` as `height * width * 3` bytes (row-major,
    /// top row first). `eye`/`target` are in the scene's own frame.
    #[pyo3(signature = (world, eye, target, fov_deg = 60.0))]
    fn render_world<'py>(
        &mut self,
        py: Python<'py>,
        world: &PyCoreWorld,
        eye: Vec3Arg,
        target: Vec3Arg,
        fov_deg: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cam = self.camera(eye, target, fov_deg)?;
        let mut out = Vec::new();
        self.render_rgb(&world.inner, &cam, &mut out)?;
        Ok(PyBytes::new(py, &out))
    }

    /// One RGB frame per env of `batch`: `len(batch) * height * width * 3`
    /// bytes. `env_indices` limits the frames to those envs, in that order.
    #[pyo3(signature = (batch, eye, target, fov_deg = 60.0, env_indices = None))]
    fn render_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch: &PyCoreWorldBatch,
        eye: Vec3Arg,
        target: Vec3Arg,
        fov_deg: f64,
        env_indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let cam = self.camera(eye, target, fov_deg)?;
        let n = batch.inner.len();
        let ids = env_indices.unwrap_or_else(|| (0..n).collect());
        let mut out = Vec::with_capacity(ids.len() * (self.width * self.height * 3) as usize);
        for i in ids {
            if i >= n {
                return Err(PyIndexError::new_err(format!("env index {i} out of range ({n})")));
            }
            self.render_rgb(batch.inner.world(i), &cam, &mut out)?;
        }
        Ok(PyBytes::new(py, &out))
    }
}

/// What GPU wgpu picks for rendering: `{"name", "backend", "device_type",
/// "driver", "driver_info"}`, or `None` when there is no usable adapter.
#[pyfunction]
pub fn gpu_info(py: Python<'_>) -> PyResult<Option<Bound<'_, PyDict>>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let Ok(adapter) = adapter else {
        return Ok(None);
    };
    let info = adapter.get_info();
    let d = PyDict::new(py);
    d.set_item("name", info.name)?;
    d.set_item("backend", format!("{:?}", info.backend))?;
    d.set_item("device_type", format!("{:?}", info.device_type))?;
    d.set_item("driver", info.driver)?;
    d.set_item("driver_info", info.driver_info)?;
    Ok(Some(d))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRenderer>()?;
    module.add_function(wrap_pyfunction!(gpu_info, module)?)?;
    Ok(())
}
