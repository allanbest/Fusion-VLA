use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use core_runtime::Pool;

#[test]
fn every_task_runs_exactly_once() {
    let pool = Pool::new(8);
    for tasks in [0, 1, 2, 7, 64, 1000] {
        let counts: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
        pool.run(tasks, |i| {
            counts[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
            "{tasks} tasks"
        );
    }
}

#[test]
fn many_small_batches() {
    // A solver issues thousands of short passes; none may be lost or doubled.
    let pool = Pool::new(6);
    let total = AtomicUsize::new(0);
    for _ in 0..20_000 {
        pool.run(5, |i| {
            total.fetch_add(i + 1, Ordering::Relaxed);
        });
    }
    assert_eq!(total.load(Ordering::Relaxed), 20_000 * 15);
}

#[test]
fn chunks_cover_the_data() {
    let pool = Pool::new(4);
    let mut data = vec![0usize; 10_001];
    pool.for_chunks_mut(&mut data, 128, |offset, part| {
        for (k, v) in part.iter_mut().enumerate() {
            *v = offset + k;
        }
    });
    assert!(data.iter().enumerate().all(|(i, &v)| i == v));
}

#[test]
fn reductions_do_not_depend_on_thread_count() {
    let values: Vec<f64> = (0..100_003)
        .map(|i| ((i as f64) * 0.37).sin() * 1e-3 + 1.0 / (1.0 + i as f64))
        .collect();
    let sum = |pool: &Pool| {
        pool.map_reduce(
            values.len(),
            1000,
            0.0,
            |r| values[r].iter().sum::<f64>(),
            |a, b| a + b,
        )
    };
    let one = sum(&Pool::new(1));
    for threads in [2, 3, 5, 8, 12] {
        assert_eq!(
            sum(&Pool::new(threads)).to_bits(),
            one.to_bits(),
            "{threads} threads"
        );
    }
}

#[test]
fn nested_calls_run_every_task_once() {
    let pool = Pool::new(4);
    let total = AtomicUsize::new(0);
    pool.run(8, |_| {
        pool.run(8, |_| {
            total.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert_eq!(total.load(Ordering::Relaxed), 64);
}

/// Islands solved side by side, each with parallel passes of its own, and
/// passes within those: every task of every batch runs exactly once, for
/// any number of threads, however many batches are open at once.
#[test]
fn batches_within_batches_lose_and_double_nothing() {
    for threads in [1, 2, 3, 8, 16] {
        let pool = Pool::new(threads);
        for round in 0..200 {
            let (outer, middle, inner) = (1 + round % 9, 1 + round % 5, 1 + round % 7);
            let counts: Vec<AtomicUsize> = (0..outer * middle * inner)
                .map(|_| AtomicUsize::new(0))
                .collect();
            pool.run(outer, |a| {
                pool.run(middle, |b| {
                    pool.run(inner, |c| {
                        counts[(a * middle + b) * inner + c].fetch_add(1, Ordering::Relaxed);
                    });
                });
            });
            assert!(
                counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "{threads} threads, round {round}"
            );
        }
    }
}

/// More batches open at once than the pool has places for: the ones that
/// find none run on their callers, and nothing is lost.
#[test]
fn more_open_batches_than_places() {
    let pool = Pool::new(8);
    let total = AtomicUsize::new(0);
    pool.run(200, |_| {
        pool.run(3, |_| {
            pool.run(3, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
    });
    assert_eq!(total.load(Ordering::Relaxed), 200 * 9);
}

/// A reduction inside a task of another batch gives the same bits as on one
/// thread: who runs which chunk never matters.
#[test]
fn nested_reductions_do_not_depend_on_thread_count() {
    let values: Vec<f64> = (0..50_003)
        .map(|i| ((i as f64) * 0.37).sin() * 1e-3 + 1.0 / (1.0 + i as f64))
        .collect();
    let sums = |pool: &Pool| {
        let mut out = vec![0.0f64; 6];
        pool.for_chunks_mut(&mut out, 1, |k, part| {
            part[0] = pool.map_reduce(
                values.len() - k,
                700,
                0.0,
                |r| values[r].iter().sum::<f64>(),
                |a, b| a + b,
            );
        });
        out
    };
    let one = sums(&Pool::new(1));
    for threads in [2, 3, 5, 8, 12] {
        let many = sums(&Pool::new(threads));
        assert!(
            one.iter()
                .zip(&many)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{threads} threads"
        );
    }
}

/// Several threads of the caller's own, each handing out batches to the
/// one pool at the same time.
#[test]
fn several_callers_at_once() {
    let pool = Pool::new(6);
    let total = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..5 {
            scope.spawn(|| {
                for _ in 0..2_000 {
                    pool.run(7, |i| {
                        total.fetch_add(i + 1, Ordering::Relaxed);
                    });
                }
            });
        }
    });
    assert_eq!(total.load(Ordering::Relaxed), 5 * 2_000 * 28);
}

/// A panic in a batch within a batch reaches the outermost caller, after
/// everything has stopped, and the pool survives it.
#[test]
fn panics_in_nested_batches_reach_the_caller() {
    let pool = Pool::new(4);
    for _ in 0..50 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(6, |a| {
                pool.run(10, |b| {
                    if a == 3 && b == 7 {
                        panic!("task 3.7");
                    }
                })
            })
        }));
        assert!(result.is_err());
    }
    let total = AtomicUsize::new(0);
    pool.run(100, |_| {
        pool.run(4, |_| {
            total.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert_eq!(total.load(Ordering::Relaxed), 400);
}

#[test]
fn panics_reach_the_caller_and_the_pool_survives() {
    let pool = Pool::new(4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(100, |i| {
            if i == 37 {
                panic!("task 37");
            }
        })
    }));
    assert!(result.is_err());
    let total = AtomicUsize::new(0);
    pool.run(100, |_| {
        total.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(total.load(Ordering::Relaxed), 100);
}

#[test]
fn borrowed_data_is_safe_to_use() {
    // The job borrows from the caller's stack, which is the point of the pool.
    let pool = Pool::new(4);
    for round in 0..100 {
        let input: Vec<usize> = (0..256).map(|i| i * round).collect();
        let output: Vec<AtomicUsize> = (0..256).map(|_| AtomicUsize::new(0)).collect();
        pool.run(256, |i| output[i].store(input[i] + 1, Ordering::Relaxed));
        assert!(output
            .iter()
            .enumerate()
            .all(|(i, o)| o.load(Ordering::Relaxed) == i * round + 1));
    }
}

/// Not a check: prints what a pass costs when every thread must take part,
/// the pool against starting scoped threads.
#[test]
fn dispatch_cost() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [2, 4, 8, cores] {
        dispatch_cost_on(threads);
    }
}

fn dispatch_cost_on(threads: usize) {
    let pool = Pool::new(threads);
    // Each task waits until all have started, so every worker must join.
    let all_join = |started: &AtomicUsize| {
        started.fetch_add(1, Ordering::AcqRel);
        while started.load(Ordering::Acquire) < threads {
            std::hint::spin_loop();
        }
    };
    let passes = 2000;
    let start = Instant::now();
    for _ in 0..passes {
        let started = AtomicUsize::new(0);
        pool.run(threads, |_| all_join(&started));
    }
    let pooled = start.elapsed().as_secs_f64() / passes as f64;
    let start = Instant::now();
    for _ in 0..passes / 10 {
        let started = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 1..threads {
                s.spawn(|| all_join(&started));
            }
            all_join(&started);
        });
    }
    let spawned = start.elapsed().as_secs_f64() / (passes / 10) as f64;
    println!(
        "one pass on {threads} threads, all taking part: pool {:.1} µs, scoped threads {:.1} µs",
        pooled * 1e6,
        spawned * 1e6
    );
}

#[test]
fn split_reduce_updates_several_arrays_in_one_pass() {
    let pool = Pool::new(4);
    let n = 10_007;
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let sum = pool.split_reduce(
        (&a[..], &mut b[..], &mut c[..]),
        500,
        0.0,
        |offset, (a, b, c)| {
            let mut s = 0.0;
            for k in 0..a.len() {
                b[k] = 2.0 * a[k];
                c[k] = (offset + k) as f64;
                s += a[k];
            }
            s
        },
        |x, y| x + y,
    );
    assert_eq!(sum, (n * (n - 1) / 2) as f64);
    assert!(b.iter().zip(&a).all(|(b, a)| *b == 2.0 * a));
    assert!(c.iter().enumerate().all(|(i, c)| *c == i as f64));
}

#[test]
fn default_threads_are_sensible() {
    let n = Pool::default_threads();
    assert!(n >= 1 && n <= std::thread::available_parallelism().unwrap().get());
    println!("default threads: {n}");
}
