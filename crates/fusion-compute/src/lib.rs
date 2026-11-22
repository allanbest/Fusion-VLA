//! Fusion L3 — a thin compute layer over wgpu.
//!
//! PLAN.md §24 specifies this layer in five rows, and they are the whole of
//! what is here: a portable abstraction over wgpu rather than three hand-kept
//! backends, command-graph batching so a step is one submission, residency
//! declared per array rather than assumed from unified memory, and readback
//! only of what something actually needs.
//!
//! ## What this crate is not
//!
//! It is not a general GPU framework, and it is deliberately small enough to
//! read in one sitting. Everything it exposes exists because a particle solver
//! needs it: a buffer of `T`, a kernel over some buffers, and a way to run a
//! sequence of kernels without stopping to talk to the CPU in between.
//!
//! ## The dependency rule, and the exception this crate is
//!
//! Fusion's L4-and-below crates have no external dependencies — see
//! `docs/STATUS.md`. A GPU abstraction cannot honour that: nobody owns a Metal
//! driver. So the rule is kept by keeping this crate *off the path*. Nothing
//! in `fusion-core`, `fusion-collide`, `fusion-solver` or `fusion-sim` depends
//! on it, a default build of the engine does not compile it, and §25's
//! headless requirement — a cluster node with no display and no GPU runs the
//! simulation — stays true by construction rather than by care.
//!
//! ## No GPU is a normal outcome
//!
//! [`Device::headless`] returns `None` where there is no adapter, and every
//! test in this crate skips rather than fails on that. CI runners frequently
//! have no GPU, and a test suite that goes red on them is a test suite people
//! learn to ignore.

#![forbid(unsafe_code)]

mod field;
mod graph;
mod kernel;

pub use field::{Field, Residency};
pub use graph::Graph;
pub use kernel::{Kernel, Params, WORKGROUP};

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// What talking to the GPU has cost so far.
///
/// PLAN.md §24: "Minimizing CPU↔GPU sync stalls is the single biggest
/// real-world GPU performance factor here." A claim like that needs a counter
/// attached to it, so this is the counter. `stalls` is the one to watch — it
/// counts the times the CPU stopped and waited for the GPU to catch up, which
/// is the thing command batching exists to avoid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Command buffers handed to the queue.
    pub submits: u64,
    /// Kernel dispatches recorded, across all submissions.
    pub dispatches: u64,
    /// Times the CPU blocked waiting for the GPU. Readbacks, mostly.
    pub stalls: u64,
    pub bytes_uploaded: u64,
    pub bytes_read_back: u64,
}

/// A compute device: an adapter, a queue, and the counters above.
pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    backend: wgpu::Backend,
    stats: Cell<Stats>,
    next_id: AtomicU64,
}

impl Device {
    /// Open a compute device with no window and no surface.
    ///
    /// `None` where there is no adapter at all. That is not an error path in
    /// the sense of something having gone wrong — it is a machine without a
    /// GPU, which §25 says has to keep working.
    pub fn headless() -> Option<Self> {
        pollster::block_on(Self::open())
    }

    async fn open() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fusion-compute"),
                required_features: wgpu::Features::empty(),
                // The downlevel baseline, with the storage-buffer limits
                // raised to what this adapter actually offers. A particle
                // solver's buffers are much larger than the 128 MiB the
                // conservative default allows, and asking for more than the
                // adapter has is a validation failure at device creation
                // rather than a fallback.
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter
                        .limits()
                        .max_storage_buffer_binding_size,
                    max_buffer_size: adapter.limits().max_buffer_size,
                    max_storage_buffers_per_shader_stage: adapter
                        .limits()
                        .max_storage_buffers_per_shader_stage,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;
        Some(Self {
            device,
            queue,
            adapter: info.name,
            backend: info.backend,
            stats: Cell::new(Stats::default()),
            next_id: AtomicU64::new(0),
        })
    }

    /// Human-readable adapter name — "Apple M2 Pro", "NVIDIA GeForce ...".
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Which graphics API this ended up on. Worth printing in any benchmark:
    /// PLAN.md §25 calls GPU behaviour divergence the real cross-platform tax,
    /// and a number taken on Metal is not a number taken on Vulkan.
    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    /// The limits this device was actually created with.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    pub fn stats(&self) -> Stats {
        self.stats.get()
    }

    pub fn reset_stats(&self) {
        self.stats.set(Stats::default());
    }

    pub(crate) fn record(&self, f: impl FnOnce(&mut Stats)) {
        let mut s = self.stats.get();
        f(&mut s);
        self.stats.set(s);
    }

    pub(crate) fn raw(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Block until the GPU has finished everything submitted so far.
    ///
    /// Counted as a stall, because it is one. Nothing in a step should call
    /// this; it is for benchmarks that need to time a dispatch rather than the
    /// act of asking for a dispatch.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        self.record(|s| s.stalls += 1);
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("adapter", &self.adapter)
            .field("backend", &self.backend)
            .field("stats", &self.stats.get())
            .finish()
    }
}
