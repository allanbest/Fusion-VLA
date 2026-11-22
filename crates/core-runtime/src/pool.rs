//! A persistent worker pool with deterministic results.
//!
//! Threads are started once and wait for work, so handing out a pass costs
//! a wake-up, not a thread start (which was a fifth of a small scene's time
//! in the main engine). Callers split work into **fixed-size chunks that do
//! not depend on the number of threads**, and reductions combine chunk
//! results in chunk order — so a result is the same to the last bit on one
//! thread or twelve.
//!
//! Several batches can be open at once: a task may hand out a batch of its
//! own (islands solved side by side, each with its own parallel passes), and
//! threads with nothing to do in one batch help with another. Which thread
//! runs which chunk never changes a result.

use std::any::Any;
use std::cell::Cell;
use std::ops::Range;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

use crate::Parts;

/// How long an idle worker keeps checking for new work before it sleeps,
/// in spin iterations (a few tens of microseconds): long enough to catch the
/// next pass of a solver loop, short enough not to burn a core between
/// frames.
const SPIN: u32 = 20_000;

/// Batches open at once: one per thread per level of nesting, so a handful.
/// A batch that finds no free slot runs on its caller.
const SLOTS: usize = 64;

/// A batch of tasks. Lives on the stack of [`Pool::run`].
struct Batch<'a> {
    job: &'a (dyn Fn(usize) + Sync),
    tasks: usize,
}

/// A place for one open batch.
struct Slot {
    /// Held by the caller whose batch is (or is about to be) here.
    taken: AtomicBool,
    /// The open batch, or null. Its lifetime is erased; see `Pool::run`.
    batch: AtomicPtr<Batch<'static>>,
    /// The next task index of the open batch.
    next: AtomicUsize,
    /// The open batch's number of tasks, for a look without entering it:
    /// only a hint of whether any are left.
    tasks: AtomicUsize,
    /// Threads inside the batch (or about to check whether one is open).
    busy: AtomicUsize,
    /// The first panic raised by a task of the batch.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

struct Shared {
    slots: [Slot; SLOTS],
    /// Bit `i` is set while slot `i` has a batch open: where to look.
    open: AtomicU64,
    /// Bumped for every batch, so that a waiting worker looks again.
    generation: AtomicU64,
    /// Workers asleep, or about to be.
    sleepers: AtomicUsize,
    stop: AtomicBool,
    /// Only for sleeping and waking.
    lock: Mutex<()>,
    wake: Condvar,
}

thread_local! {
    /// How many pool tasks this thread is inside of, one within another.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// A fixed set of worker threads.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool of `threads` threads in all: the caller and `threads − 1`
    /// workers. One thread means everything runs on the caller.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            slots: std::array::from_fn(|_| Slot {
                taken: AtomicBool::new(false),
                batch: AtomicPtr::new(std::ptr::null_mut()),
                next: AtomicUsize::new(0),
                tasks: AtomicUsize::new(0),
                busy: AtomicUsize::new(0),
                panic: Mutex::new(None),
            }),
            open: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            sleepers: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("fusion-worker-{i}"))
                    .spawn(move || work(&shared))
                    .expect("starting a worker thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// A pool with [`Pool::default_threads`] threads.
    pub fn with_default_threads() -> Self {
        Self::new(Self::default_threads())
    }

    /// A pool shared by the whole process, of [`Pool::default_threads`].
    pub fn shared() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(Pool::with_default_threads)
    }

    /// The `FUSION_THREADS` environment variable if set; otherwise the
    /// performance cores where the machine has slower efficiency cores too
    /// (one straggling chunk on a slow core holds up a whole pass);
    /// otherwise every core.
    pub fn default_threads() -> usize {
        if let Some(n) = std::env::var("FUSION_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return n.max(1);
        }
        let all = std::thread::available_parallelism().map_or(1, |n| n.get());
        performance_cores().map_or(all, |p| p.clamp(1, all))
    }

    /// Threads in all, the caller included.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `job(i)` for every `i` in `0..tasks`, spread over the threads,
    /// and returns when all have finished. A panic in any task is raised
    /// again here, after every task has stopped.
    ///
    /// A task may call this itself: its batch is open beside the one it
    /// belongs to, and idle threads help with either.
    pub fn run(&self, tasks: usize, job: impl Fn(usize) + Sync) {
        let shared = &*self.shared;
        let claimed = (tasks > 1 && !self.workers.is_empty())
            .then(|| {
                shared.slots.iter().position(|slot| {
                    slot.taken
                        .compare_exchange(false, true, SeqCst, SeqCst)
                        .is_ok()
                })
            })
            .flatten();
        let Some(index) = claimed else {
            (0..tasks).for_each(job);
            return;
        };
        let slot = &shared.slots[index];
        let batch = Batch { job: &job, tasks };
        // SAFETY: only the lifetime is erased. A thread announces itself in
        // the slot's `busy` *before* it reads the slot's `batch`, and the
        // caller clears `batch` *before* it waits for `busy` to reach zero;
        // with sequentially consistent order on both, either the thread
        // sees the batch cleared and never touches it, or the caller sees
        // the thread and waits for it. So nothing reads `batch` or `job`
        // after this function returns or unwinds. The slot is the caller's
        // (`taken`) from before the batch is set until after that wait, so
        // a thread that arrives later finds it empty or holding a later
        // caller's batch, under that caller's same guarantee.
        let pointer = &batch as *const Batch<'_> as *mut Batch<'static>;
        slot.next.store(0, SeqCst);
        slot.tasks.store(tasks, SeqCst);
        slot.batch.store(pointer, SeqCst);
        shared.open.fetch_or(1 << index, SeqCst);
        shared.generation.fetch_add(1, SeqCst);
        if shared.sleepers.load(SeqCst) > 0 {
            let _lock = shared.lock.lock().unwrap();
            shared.wake.notify_all();
        }

        take_tasks(slot, &batch);

        shared.open.fetch_and(!(1 << index), SeqCst);
        slot.tasks.store(0, SeqCst);
        slot.batch.store(std::ptr::null_mut(), SeqCst);
        // The caller of an outermost batch has nothing else to do while the
        // last tasks finish on other threads, and those may have batches of
        // their own open: it helps with them, a task at a time. A caller
        // that is itself inside a task only waits (its batch's last tasks
        // are short, and taking on another's would bury its own under them).
        let outermost = DEPTH.with(Cell::get) == 0;
        let mut spins = 0u32;
        while slot.busy.load(SeqCst) != 0 {
            if outermost && help_once(shared) {
                spins = 0;
            } else if spins < SPIN {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        let panic = slot.panic.lock().unwrap().take();
        slot.taken.store(false, SeqCst);
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    /// Runs `body` on consecutive ranges of `0..n`, each `chunk` long (the
    /// last may be shorter).
    pub fn for_ranges(&self, n: usize, chunk: usize, body: impl Fn(Range<usize>) + Sync) {
        let chunk = chunk.max(1);
        self.run(n.div_ceil(chunk), |c| {
            body(c * chunk..((c + 1) * chunk).min(n))
        });
    }

    /// Runs `body(offset, part)` on consecutive parts of `data`, each `chunk`
    /// long (the last may be shorter), `offset` being the part's start.
    pub fn for_chunks_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        body: impl Fn(usize, &mut [T]) + Sync,
    ) {
        self.split_reduce(data, chunk, (), body, |_, _| ());
    }

    /// `reduce(…reduce(reduce(identity, map(r₀)), map(r₁))…, map(rₖ))` over
    /// consecutive ranges `rᵢ` of `0..n`, each `chunk` long: the maps run in
    /// parallel, the reduction in order, so the result does not depend on
    /// the number of threads.
    pub fn map_reduce<T: Send>(
        &self,
        n: usize,
        chunk: usize,
        identity: T,
        map: impl Fn(Range<usize>) -> T + Sync,
        reduce: impl Fn(T, T) -> T,
    ) -> T {
        let chunk = chunk.max(1);
        let chunks = n.div_ceil(chunk);
        if chunks <= 1 {
            return if n == 0 {
                identity
            } else {
                reduce(identity, map(0..n))
            };
        }
        let results: Vec<Mutex<Option<T>>> = (0..chunks).map(|_| Mutex::new(None)).collect();
        self.run(chunks, |c| {
            let value = map(c * chunk..((c + 1) * chunk).min(n));
            *results[c].lock().unwrap() = Some(value);
        });
        results.into_iter().fold(identity, |acc, r| {
            reduce(acc, r.into_inner().unwrap().unwrap())
        })
    }

    /// Cuts `data` — a slice, or a tuple of equally long slices — into
    /// consecutive parts of `chunk` (the last may be shorter), runs
    /// `map(offset, part)` on each in parallel, and reduces the results in
    /// order, as [`Pool::map_reduce`]. One pass can so update several arrays
    /// and return a sum.
    pub fn split_reduce<P: Parts, R: Send>(
        &self,
        data: P,
        chunk: usize,
        identity: R,
        map: impl Fn(usize, P) -> R + Sync,
        reduce: impl Fn(R, R) -> R,
    ) -> R {
        let chunk = chunk.max(1);
        let n = data.len();
        if n <= chunk {
            return if n == 0 {
                identity
            } else {
                reduce(identity, map(0, data))
            };
        }
        let mut parts = Vec::with_capacity(n.div_ceil(chunk));
        let mut rest = data;
        while rest.len() > chunk {
            let (part, tail) = rest.split_at(chunk);
            parts.push(Mutex::new(Some(part)));
            rest = tail;
        }
        parts.push(Mutex::new(Some(rest)));
        let results: Vec<Mutex<Option<R>>> = (0..parts.len()).map(|_| Mutex::new(None)).collect();
        self.run(parts.len(), |c| {
            let part = parts[c].lock().unwrap().take().unwrap();
            *results[c].lock().unwrap() = Some(map(c * chunk, part));
        });
        results.into_iter().fold(identity, |acc, r| {
            reduce(acc, r.into_inner().unwrap().unwrap())
        })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        self.shared.generation.fetch_add(1, SeqCst);
        {
            let _lock = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Runs task `i` of `batch`, keeping a panic for the batch's caller.
fn run_task(slot: &Slot, batch: &Batch, i: usize) {
    DEPTH.with(|d| d.set(d.get() + 1));
    let result = catch_unwind(AssertUnwindSafe(|| (batch.job)(i)));
    DEPTH.with(|d| d.set(d.get() - 1));
    if let Err(panic) = result {
        // Stop handing out the rest of the batch, and keep the first panic.
        slot.next.store(usize::MAX / 2, Relaxed);
        slot.panic.lock().unwrap().get_or_insert(panic);
    }
}

/// Claims and runs tasks of `batch`, the batch open in `slot`, until none
/// are left.
fn take_tasks(slot: &Slot, batch: &Batch) {
    loop {
        let i = slot.next.fetch_add(1, Relaxed);
        if i >= batch.tasks {
            return;
        }
        run_task(slot, batch, i);
    }
}

/// Whether `slot` seems to have tasks nobody has claimed. A look from
/// outside the batch: it may be out of date either way, which costs a
/// wasted visit or a missed one (the next batch brings everyone back).
fn waiting(slot: &Slot) -> bool {
    slot.next.load(Relaxed) < slot.tasks.load(Relaxed)
}

/// Runs one unclaimed task of any open batch, if there is one.
fn help_once(shared: &Shared) -> bool {
    let mut open = shared.open.load(SeqCst);
    while open != 0 {
        let slot = &shared.slots[open.trailing_zeros() as usize];
        open &= open - 1;
        if !waiting(slot) {
            continue;
        }
        slot.busy.fetch_add(1, SeqCst);
        let pointer = slot.batch.load(SeqCst);
        let mut ran = false;
        if !pointer.is_null() {
            // SAFETY: `busy` was raised before the load; see `Pool::run`.
            let batch = unsafe { &*pointer };
            let i = slot.next.fetch_add(1, Relaxed);
            if i < batch.tasks {
                run_task(slot, batch, i);
                ran = true;
            }
        }
        slot.busy.fetch_sub(1, SeqCst);
        if ran {
            return true;
        }
    }
    false
}

/// Works through every open batch that has unclaimed tasks, earliest slot
/// first, until none has.
fn help(shared: &Shared) {
    loop {
        let mut helped = false;
        let mut open = shared.open.load(SeqCst);
        while open != 0 {
            let slot = &shared.slots[open.trailing_zeros() as usize];
            open &= open - 1;
            if !waiting(slot) {
                continue;
            }
            slot.busy.fetch_add(1, SeqCst);
            let pointer = slot.batch.load(SeqCst);
            if !pointer.is_null() {
                // SAFETY: `busy` was raised before the load; see `Pool::run`.
                take_tasks(slot, unsafe { &*pointer });
                helped = true;
            }
            slot.busy.fetch_sub(1, SeqCst);
        }
        if !helped {
            return;
        }
    }
}

fn work(shared: &Shared) {
    let mut seen = 0;
    loop {
        // Wait for a new generation: spin first, then sleep.
        let mut spins = 0;
        while shared.generation.load(SeqCst) == seen {
            if spins < SPIN {
                std::hint::spin_loop();
                spins += 1;
                continue;
            }
            let lock = shared.lock.lock().unwrap();
            shared.sleepers.fetch_add(1, SeqCst);
            // Checked again after announcing the sleep, so that a batch
            // posted in between is not missed (see `Pool::run`).
            let _lock = shared
                .wake
                .wait_while(lock, |_| shared.generation.load(SeqCst) == seen)
                .unwrap();
            shared.sleepers.fetch_sub(1, SeqCst);
        }
        if shared.stop.load(SeqCst) {
            return;
        }
        // Read before looking, so a batch opened while looking is a new
        // generation and is looked for again.
        seen = shared.generation.load(SeqCst);
        help(shared);
    }
}

/// Performance cores on a machine that also has efficiency cores.
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}
