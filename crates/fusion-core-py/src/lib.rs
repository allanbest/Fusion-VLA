//! The native half of the `fusion_vla` Python package. Everything lives in
//! `core.rs`; Python imports it as `fusion_vla.core` (re-exported from
//! `fusion_vla._core`).

use pyo3::prelude::*;

mod core;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(module)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
