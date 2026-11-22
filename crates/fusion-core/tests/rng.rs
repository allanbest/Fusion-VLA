//! The generator, against the properties a reproducibility seed needs it to
//! have. Not a statistical test suite — PCG's distribution is not what is in
//! doubt here, the wiring is.

use fusion_core::Rng;
use fusion_core::World;

#[test]
fn the_same_seed_gives_the_same_stream() {
    let mut a = Rng::new(12345);
    let mut b = Rng::new(12345);
    for _ in 0..1000 {
        assert_eq!(a.next_u32(), b.next_u32());
    }
}

#[test]
fn different_seeds_give_different_streams() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    let differences = (0..64).filter(|_| a.next_u32() != b.next_u32()).count();
    assert_eq!(differences, 64, "consecutive seeds produced matching draws");
}

#[test]
fn consecutive_seeds_are_not_correlated() {
    // The reason for choosing PCG over a bare shift-register generator. A
    // sweep over `seed = 0..n` is the ordinary way to run one, and a generator
    // whose first outputs track the seed makes every run in the sweep a
    // near-copy of its neighbour.
    let firsts: Vec<f32> = (0..64u64).map(|s| Rng::new(s).next_f32()).collect();
    // Spread across the unit interval rather than clustered near zero.
    let mean = firsts.iter().sum::<f32>() / firsts.len() as f32;
    assert!(
        (mean - 0.5).abs() < 0.12,
        "mean of first draws across 64 seeds is {mean}"
    );
    // And not monotone in the seed, which is what a naive LCG's low bits do.
    let ascending = firsts.windows(2).filter(|w| w[1] > w[0]).count();
    assert!(
        (10..53).contains(&ascending),
        "{ascending} of 63 consecutive pairs ascended"
    );
}

#[test]
fn streams_are_independent() {
    let mut a = Rng::with_stream(7, 1);
    let mut b = Rng::with_stream(7, 2);
    let same = (0..64).filter(|_| a.next_u32() == b.next_u32()).count();
    assert_eq!(same, 0, "two streams of one seed produced matching draws");
}

#[test]
fn floats_stay_in_the_half_open_unit_interval() {
    let mut rng = Rng::new(99);
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for _ in 0..200_000 {
        let v = rng.next_f32();
        assert!((0.0..1.0).contains(&v), "produced {v}");
        low = low.min(v);
        high = high.max(v);
    }
    assert!(low < 0.001 && high > 0.999, "range was [{low}, {high}]");
}

#[test]
fn below_is_in_range_and_not_biased_towards_small_values() {
    let mut rng = Rng::new(4);
    // A deliberately awkward bound: not a power of two, so a bare modulo
    // would show its bias.
    let n = 7u32;
    let mut counts = [0u32; 7];
    for _ in 0..70_000 {
        let v = rng.below(n);
        assert!(v < n);
        counts[v as usize] += 1;
    }
    for (value, count) in counts.iter().enumerate() {
        assert!(
            (9_000..11_000).contains(count),
            "value {value} came up {count} times in 70000, expected about 10000"
        );
    }
    assert_eq!(rng.below(0), 0);
    assert_eq!(rng.below(1), 0);
}

#[test]
fn points_in_the_unit_sphere_are_in_the_unit_sphere() {
    let mut rng = Rng::new(2024);
    let mut max = 0.0f32;
    for _ in 0..20_000 {
        let p = rng.in_unit_sphere();
        let r = p.length();
        assert!(r <= 1.0, "produced a point at radius {r}");
        max = max.max(r);
    }
    assert!(max > 0.99, "never approached the surface: max radius {max}");
}

#[test]
fn a_world_hands_out_generators_seeded_from_itself() {
    // PLAN.md §22: seeds explicit, never implicit. Reaching a generator
    // through the world is what makes that structural rather than a
    // convention — there is no other way to get one that a project file has
    // not recorded.
    let world = World::new().with_seed(0xabc_def);
    assert_eq!(world.seed, 0xabc_def);

    let mut a = world.rng(0);
    let mut b = world.rng(0);
    assert_eq!(a.next_u64(), b.next_u64());

    let mut other = world.rng(1);
    assert_ne!(world.rng(0).next_u64(), other.next_u64());

    // A different world seed gives a different stream on the same channel.
    let elsewhere = World::new().with_seed(0xabc_de0);
    assert_ne!(world.rng(0).next_u64(), elsewhere.rng(0).next_u64());
}
