//! Layer 0 of the Fusion core: the numbers everything else is written in.
//!
//! Everything is `f64` and in SI units — metres, kilograms, seconds, kelvin,
//! newtons, pascals — so that a quantity's unit is never a convention of the
//! code that holds it. Names and documentation say the unit; the type does
//! not, which keeps the arithmetic free.

pub mod constants;
mod mat3;
mod pose;
mod quat;
mod rng;
mod vec3;

pub use mat3::Mat3;
pub use pose::Pose;
pub use quat::Quat;
pub use rng::Rng;
pub use vec3::Vec3;

/// Shorthand for [`Vec3::new`].
#[inline]
pub const fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x, y, z)
}
