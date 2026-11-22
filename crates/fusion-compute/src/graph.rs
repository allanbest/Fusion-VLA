//! Command-graph batching: a step's worth of dispatches, submitted once.
//!
//! PLAN.md §24 puts this first among GPU concerns — "build the DAG of
//! dispatches per step, submit once. Minimizing CPU↔GPU sync stalls is the
//! single biggest real-world GPU performance factor here." The reason is that
//! a dispatch is cheap and a *submission* is not: each one crosses into the
//! driver, and a step that submits twelve times and waits for each has spent
//! its frame on twelve round trips rather than on physics.
//!
//! ## What the DAG is for, and what it is not for
//!
//! It would be easy to overstate this. WebGPU already orders dispatches within
//! a compute pass and inserts the memory barriers between them, so a graph
//! cannot make consecutive kernels overlap and does not try to. Two things it
//! does do, and both are worth the type:
//!
//! - **One submission.** Every dispatch in the graph goes into one encoder and
//!   one pass, and the queue sees one command buffer. Measured against the
//!   same kernels submitted one at a time in [`Graph::submit_each`], which
//!   exists so the benchmark can show the difference rather than assert it.
//! - **The declared reads and writes are checked.** A kernel that writes a
//!   CPU-authoritative field is rejected before it runs, and the dependency
//!   *depth* — how much of the step is inherently sequential — is reported,
//!   because that is the number which says whether a bigger GPU would help.
//!
//! What it deliberately does not do is reorder. Declaration order is already a
//! topological order of the dependencies, and shuffling independent dispatches
//! within a pass buys nothing in this execution model while making a
//! bit-identical replay harder to reason about (§34).

use crate::field::AnyField;
use crate::{Device, Kernel, Params, Residency};

/// One recorded dispatch.
struct Step<'a> {
    kernel: &'a Kernel,
    binds: Vec<(&'a dyn AnyField, bool)>,
    params: Params,
    items: usize,
}

/// One recorded operation: a dispatch, or a buffer clear.
///
/// Clears are here rather than on the device because a histogram has to be
/// zeroed at the top of every substep, and a `Device::clear` would be its own
/// submission — which is precisely the per-step driver traffic this module
/// exists to remove. `clear_buffer` cannot be recorded inside a compute pass,
/// so a graph with clears in the middle becomes several passes inside one
/// encoder. Still one submission, which is the property that matters.
enum Op<'a> {
    Dispatch(Step<'a>),
    Clear(&'a dyn AnyField),
}

/// A step's worth of dispatches, recorded and submitted together.
///
/// ```ignore
/// let mut graph = Graph::new("fluid substep");
/// graph.step(&gravity).write(&velocity).items(n);
/// graph.step(&predict).read(&velocity).write(&position).items(n);
/// graph.submit(&device);
/// ```
pub struct Graph<'a> {
    label: String,
    ops: Vec<Op<'a>>,
}

impl<'a> Graph<'a> {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ops: Vec::new(),
        }
    }

    /// Zero a field, in order, as part of this graph.
    pub fn clear<T: bytemuck::Pod>(&mut self, field: &'a crate::Field<T>) {
        self.ops.push(Op::Clear(field));
    }

    /// The dispatches, in order, skipping the clears.
    fn steps(&self) -> impl Iterator<Item = &Step<'a>> {
        self.ops.iter().filter_map(|op| match op {
            Op::Dispatch(step) => Some(step),
            Op::Clear(_) => None,
        })
    }

    /// Begin recording a dispatch. Finish it with [`StepBuilder::items`].
    #[must_use = "a step is only recorded when `items` is called"]
    pub fn step<'g>(&'g mut self, kernel: &'a Kernel) -> StepBuilder<'g, 'a> {
        StepBuilder {
            graph: self,
            step: Step {
                kernel,
                binds: Vec::new(),
                params: Params::default(),
                items: 0,
            },
        }
    }

    /// Dispatches recorded. Clears are not dispatches and are not counted.
    pub fn len(&self) -> usize {
        self.steps().count()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// How many dispatches deep the dependency chain is.
    ///
    /// Steps that share a level read and write disjoint fields, so nothing
    /// orders them against each other. The depth is therefore the number of
    /// stages the step *cannot* avoid running in sequence, and it is the
    /// ceiling on what any amount of extra hardware can do for it — a
    /// twelve-dispatch step with a depth of twelve is a latency problem, and
    /// one with a depth of three is a throughput problem.
    pub fn depth(&self) -> usize {
        let steps: Vec<&Step<'a>> = self.steps().collect();
        let mut levels: Vec<usize> = Vec::with_capacity(steps.len());
        for (i, step) in steps.iter().enumerate() {
            let mut level = 0;
            for (j, earlier) in steps[..i].iter().enumerate() {
                if conflicts(earlier, step) {
                    level = level.max(levels[j] + 1);
                }
            }
            levels.push(level);
        }
        levels.iter().map(|l| l + 1).max().unwrap_or(0)
    }

    /// Every residency and binding rule the recorded steps break.
    ///
    /// Returned rather than panicked so a caller can report all of them at
    /// once; [`Graph::submit`] panics on the first, which is what a solver
    /// wants.
    pub fn violations(&self) -> Vec<String> {
        let mut out = Vec::new();
        for step in self.steps() {
            if step.binds.len() != step.kernel.buffers() {
                out.push(format!(
                    "`{}` binds {} buffers, {} given",
                    step.kernel.label(),
                    step.kernel.buffers(),
                    step.binds.len()
                ));
            }
            for (field, written) in &step.binds {
                if *written && field.residency() == Residency::Cpu {
                    out.push(format!(
                        "`{}` writes `{}`, which is CPU-authoritative",
                        step.kernel.label(),
                        field.label()
                    ));
                }
            }
        }
        out
    }

    /// Record every dispatch into one command buffer and submit it once.
    ///
    /// One encoder, one compute pass, one submission, no wait. The GPU is
    /// still working when this returns, which is the point: the CPU goes back
    /// to whatever else the step has to do, and nothing blocks until something
    /// actually asks to read a result.
    pub fn submit(&self, device: &Device) {
        self.run(device, true);
    }

    /// The same dispatches, one submission each.
    ///
    /// This is the thing [`Graph::submit`] is faster than, and it is in the
    /// library rather than in the benchmark so that the comparison is between
    /// two code paths that are otherwise identical. Not for production use.
    pub fn submit_each(&self, device: &Device) {
        self.run(device, false);
    }

    fn run(&self, device: &Device, batched: bool) {
        if let Some(problem) = self.violations().first() {
            panic!("{}: {problem}", self.label);
        }
        if self.ops.is_empty() {
            return;
        }
        let steps: Vec<&Step<'a>> = self.steps().collect();

        // One uniform buffer for the whole graph, with each dispatch's
        // parameters at its own dynamic offset. The alternative — a buffer per
        // dispatch — allocates once per kernel per frame, which is exactly the
        // per-step driver traffic this module exists to remove.
        let stride = align_to(
            std::mem::size_of::<Params>() as u32,
            device
                .raw()
                .limits()
                .min_uniform_buffer_offset_alignment
                .max(1),
        );
        let uniform = if steps.is_empty() {
            None
        } else {
            let mut bytes = vec![0u8; stride as usize * steps.len()];
            for (i, step) in steps.iter().enumerate() {
                let at = i * stride as usize;
                bytes[at..at + std::mem::size_of::<Params>()]
                    .copy_from_slice(bytemuck::bytes_of(&step.params));
            }
            let buffer = device.raw().create_buffer(&wgpu::BufferDescriptor {
                label: Some("graph params"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            device.queue().write_buffer(&buffer, 0, &bytes);
            Some(buffer)
        };

        let groups: Vec<wgpu::BindGroup> = steps
            .iter()
            .map(|step| {
                let uniform = uniform.as_ref().expect("a dispatch implies a uniform");
                let mut entries: Vec<wgpu::BindGroupEntry> = step
                    .binds
                    .iter()
                    .enumerate()
                    .map(|(i, (field, _))| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: field.buffer().as_entire_binding(),
                    })
                    .collect();
                entries.push(wgpu::BindGroupEntry {
                    binding: step.binds.len() as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                    }),
                });
                device.raw().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(step.kernel.label()),
                    layout: step.kernel.layout(),
                    entries: &entries,
                })
            })
            .collect();

        // Consecutive dispatches share a compute pass; a clear ends the pass
        // it lands in and the next dispatch opens another. Passes are cheap
        // inside one encoder — it is the submission that is not.
        let encode = |encoder: &mut wgpu::CommandEncoder, ops: &[(usize, &Op<'a>)]| {
            let mut at = 0usize;
            while at < ops.len() {
                if let Op::Clear(field) = ops[at].1 {
                    encoder.clear_buffer(field.buffer(), 0, None);
                    at += 1;
                    continue;
                }
                // A run of consecutive dispatches shares one pass. The pass
                // has to be scoped so it is dropped before the encoder is
                // needed again for the clear that ended the run.
                let from = at;
                while at < ops.len() && matches!(ops[at].1, Op::Dispatch(_)) {
                    at += 1;
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&self.label),
                    timestamp_writes: None,
                });
                for (index, op) in &ops[from..at] {
                    let Op::Dispatch(step) = op else { continue };
                    pass.set_pipeline(step.kernel.pipeline());
                    pass.set_bind_group(0, &groups[*index], &[*index as u32 * stride]);
                    pass.dispatch_workgroups(step.workgroups(device), 1, 1);
                }
            }
        };

        // Dispatch index, for the bind group and uniform offset, alongside the
        // op — clears do not consume one.
        let mut numbered: Vec<(usize, &Op<'a>)> = Vec::with_capacity(self.ops.len());
        let mut next = 0usize;
        for op in &self.ops {
            let index = match op {
                Op::Dispatch(_) => {
                    next += 1;
                    next - 1
                }
                Op::Clear(_) => usize::MAX,
            };
            numbered.push((index, op));
        }

        if batched {
            let mut encoder =
                device
                    .raw()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&self.label),
                    });
            encode(&mut encoder, &numbered);
            device.queue().submit(Some(encoder.finish()));
            device.record(|s| {
                s.submits += 1;
                s.dispatches += steps.len() as u64;
            });
        } else {
            for op in &numbered {
                let mut encoder =
                    device
                        .raw()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&self.label),
                        });
                encode(&mut encoder, std::slice::from_ref(op));
                device.queue().submit(Some(encoder.finish()));
                device.record(|s| {
                    s.submits += 1;
                    if matches!(op.1, Op::Dispatch(_)) {
                        s.dispatches += 1;
                    }
                });
            }
        }
    }
}

impl Step<'_> {
    /// Workgroups to dispatch, and the check that the count is legal.
    fn workgroups(&self, device: &Device) -> u32 {
        let wanted = self.items.div_ceil(self.kernel.workgroup() as usize).max(1) as u64;
        let limit = device.limits().max_compute_workgroups_per_dimension as u64;
        if wanted <= limit {
            return wanted as u32;
        }
        assert!(
            self.kernel.is_strided(),
            "`{}` wants {wanted} workgroups over {} elements and the limit is \
             {limit}; either dispatch less or declare the kernel `.strided()`",
            self.kernel.label(),
            self.items
        );
        limit as u32
    }
}

/// True where `later` has to see what `earlier` did: a read of something
/// written, or two writes to the same field.
fn conflicts(earlier: &Step<'_>, later: &Step<'_>) -> bool {
    earlier.binds.iter().any(|(a, a_writes)| {
        later
            .binds
            .iter()
            .any(|(b, b_writes)| a.id() == b.id() && (*a_writes || *b_writes))
    })
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

/// A dispatch under construction. See [`Graph::step`].
#[must_use = "a step is only recorded when `items` is called"]
pub struct StepBuilder<'g, 'a> {
    graph: &'g mut Graph<'a>,
    step: Step<'a>,
}

impl<'a> StepBuilder<'_, 'a> {
    /// Bind a field the kernel only reads.
    pub fn read<T: bytemuck::Pod>(mut self, field: &'a crate::Field<T>) -> Self {
        self.step.binds.push((field, false));
        self
    }

    /// Bind a field the kernel writes. Must be GPU-authoritative.
    pub fn write<T: bytemuck::Pod>(mut self, field: &'a crate::Field<T>) -> Self {
        self.step.binds.push((field, true));
        self
    }

    pub fn params(mut self, params: Params) -> Self {
        self.step.params = params;
        self
    }

    /// Record the dispatch, over `items` threads.
    ///
    /// Threads, not elements — they are the same number for every kernel that
    /// handles one element each, which is most of them, and they are not for a
    /// scan whose threads take four apiece.
    ///
    /// The element count also becomes [`Params::count`] unless `params` has
    /// already set one, so the common case does not have to say it twice.
    pub fn items(mut self, items: usize) {
        self.step.items = items;
        if self.step.params.count == 0 {
            self.step.params.count = items as u32;
        }
        self.graph.ops.push(Op::Dispatch(self.step));
    }
}
