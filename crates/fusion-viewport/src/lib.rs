//! Fusion L6 — the debug viewport.
//!
//! PLAN.md §18 puts a debug viewport in v0.1 and calls it mandatory: you
//! cannot build a physics engine without being able to see it. It also draws
//! the boundary this crate must respect — a good viewport is not a renderer,
//! and Fusion exports to Blender or Houdini rather than growing into one.
//!
//! The dependency direction is the important architectural property here.
//! This crate depends on `fusion-core` to read shapes and poses. **Nothing in
//! `fusion-core`, `fusion-collide`, `fusion-solver` or `fusion-sim` may depend
//! on this crate**, because the research workflow in §23 lives on machines
//! with no display at all.

#![deny(unsafe_code)]

pub mod app;
pub mod camera;
pub mod colormap;
pub mod debug_draw;
pub mod mesh;
pub mod renderer;
pub mod screenshot;
pub mod volume;

pub use app::{run, Scene, ViewportConfig};
pub use camera::OrbitCamera;
pub use colormap::{blackbody, Colormap, ScalarField};
pub use debug_draw::{cloth, pinned_particles, DrawStyle};
pub use mesh::{Geometry, Surface, Vertex};
pub use renderer::{Image, Overlay, RenderSettings, Renderer};
pub use screenshot::{
    encode_png, render_offscreen, render_offscreen_with, save_png, try_render_offscreen,
};
pub use volume::{Volume, MAX_DIM};
