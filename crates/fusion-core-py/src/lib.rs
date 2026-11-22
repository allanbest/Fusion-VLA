//! The native half of the `fusion_vla` Python package. Python imports it as
//! `fusion_vla.core` (re-exported from `fusion_vla._core`).
//!
//! * `core.rs`   — Fusion Core: worlds, mechanics, batches, adjoints.
//! * `render.rs` — GPU camera rendering (feature `render`, on by default).

use pyo3::prelude::*;

mod core;
#[cfg(feature = "render")]
mod render;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(module)?;
    #[cfg(feature = "render")]
    render::register(module)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("HAS_RENDER", cfg!(feature = "render"))?;
    Ok(())
}
