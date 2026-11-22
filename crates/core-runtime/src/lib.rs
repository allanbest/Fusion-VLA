//! Layer 1 of the Fusion core: how work is run.
//!
//! - [`Pool`]: persistent worker threads, with splitting and reductions whose
//!   results do not depend on the number of threads.
//! - [`Parts`]: slices and tuples of slices that split for threads.
//! - [`Profile`]: named timers.
//!
//! Arenas and GPU queues belong here when something needs them.

mod parts;
mod pool;
mod profile;

pub use parts::Parts;
pub use pool::Pool;
pub use profile::{Entry, Profile};
