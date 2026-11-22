//! The broad phase is tested against brute force, because a missed pair is a
//! silent failure: nothing crashes, objects simply pass through each other.

use fusion_collide::broad::{find_pairs_brute_force, SweepAndPrune};
use fusion_math::{vec3, Aabb, Vec3};

/// Deterministic LCG. Tests that depend on the platform RNG are not tests.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

fn random_boxes(n: usize, seed: u64, spread: f32) -> Vec<Aabb> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let c = vec3(
                rng.range(-spread, spread),
                rng.range(-spread, spread),
                rng.range(-spread, spread),
            );
            let h = vec3(
                rng.range(0.2, 1.5),
                rng.range(0.2, 1.5),
                rng.range(0.2, 1.5),
            );
            Aabb::from_center_half_extents(c, h)
        })
        .collect()
}

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

#[test]
fn sweep_and_prune_finds_exactly_the_brute_force_pairs() {
    for (seed, spread) in [(1u64, 3.0f32), (7, 8.0), (99, 20.0)] {
        let aabbs = random_boxes(200, seed, spread);
        let dynamic = vec![true; aabbs.len()];

        let mut expected = Vec::new();
        find_pairs_brute_force(&aabbs, &dynamic, &mut expected);

        let mut sap = SweepAndPrune::new();
        let mut got = Vec::new();
        sap.find_pairs(&aabbs, &dynamic, &mut got);

        assert_eq!(
            sorted(got),
            sorted(expected),
            "seed {seed}, spread {spread}"
        );
    }
}

#[test]
fn incremental_updates_stay_correct_across_frames() {
    // The sorted order persists between frames; the point of that is speed, but
    // it must not cost correctness once things have moved a long way.
    let mut aabbs = random_boxes(120, 42, 6.0);
    let dynamic = vec![true; aabbs.len()];
    let mut sap = SweepAndPrune::new();
    let mut got = Vec::new();
    let mut expected = Vec::new();

    for frame in 0..30 {
        for (i, b) in aabbs.iter_mut().enumerate() {
            let drift = vec3(
                0.13 * (i % 7) as f32 - 0.4,
                -0.05,
                0.07 * (i % 5) as f32 - 0.15,
            );
            *b = Aabb::new(b.min + drift, b.max + drift);
        }
        sap.find_pairs(&aabbs, &dynamic, &mut got);
        find_pairs_brute_force(&aabbs, &dynamic, &mut expected);
        assert_eq!(
            sorted(got.clone()),
            sorted(expected.clone()),
            "frame {frame}"
        );
    }
}

#[test]
fn static_pairs_are_skipped() {
    let aabbs = vec![
        Aabb::from_center_half_extents(Vec3::ZERO, Vec3::ONE),
        Aabb::from_center_half_extents(vec3(0.5, 0.0, 0.0), Vec3::ONE),
        Aabb::from_center_half_extents(vec3(1.0, 0.0, 0.0), Vec3::ONE),
    ];
    let dynamic = vec![false, false, true];
    let mut sap = SweepAndPrune::new();
    let mut got = Vec::new();
    sap.find_pairs(&aabbs, &dynamic, &mut got);
    let got = sorted(got);
    assert!(
        !got.contains(&(0, 1)),
        "two static bodies are never worth a narrow-phase call"
    );
    assert!(got.contains(&(0, 2)) && got.contains(&(1, 2)));
}

// --- the grid, against the same reference -------------------------------

/// The grid finds exactly the brute-force pairs, over scenes chosen to break
/// it: uniform clouds, one enormous body among small ones, mixed sizes, and
/// bodies far from the origin.
#[test]
fn the_grid_finds_exactly_the_brute_force_pairs() {
    for (name, aabbs) in grid_scenes() {
        let dynamic = vec![true; aabbs.len()];
        let mut expected = Vec::new();
        find_pairs_brute_force(&aabbs, &dynamic, &mut expected);
        expected.sort_unstable();

        let mut grid = fusion_collide::Grid::new();
        let mut got = Vec::new();
        grid.find_pairs(&aabbs, &dynamic, &mut got);
        assert_eq!(got, expected, "{name}: the grid missed or invented pairs");
    }
}

/// The two structures produce *identical* lists, not equivalent ones.
///
/// This is what makes the choice between them invisible to physics. The solver
/// is Gauss-Seidel: constraints arriving in a different order give a different
/// answer, so a scene that switched structures at some body count would change
/// behaviour for no reason a person could see.
#[test]
fn the_grid_and_the_sweep_agree_pair_for_pair() {
    for (name, aabbs) in grid_scenes() {
        let dynamic: Vec<bool> = (0..aabbs.len()).map(|i| i % 7 != 0).collect();

        let mut sap = SweepAndPrune::new();
        let mut swept = Vec::new();
        sap.find_pairs(&aabbs, &dynamic, &mut swept);
        swept.sort_unstable();

        let mut grid = fusion_collide::Grid::new();
        let mut gridded = Vec::new();
        grid.find_pairs(&aabbs, &dynamic, &mut gridded);

        assert_eq!(gridded, swept, "{name}: the two broad phases disagree");
    }
}

/// A pair that shares several cells is still one pair.
#[test]
fn the_grid_reports_each_pair_once() {
    // Two large-ish boxes overlapping across a swathe of cells, among small
    // ones that set the cell size.
    let mut aabbs = random_boxes(600, 11, 6.0);
    aabbs.push(Aabb::new(vec3(-1.5, -1.5, -1.5), vec3(1.5, 1.5, 1.5)));
    aabbs.push(Aabb::new(vec3(-1.4, -1.4, -1.4), vec3(1.6, 1.6, 1.6)));
    let dynamic = vec![true; aabbs.len()];

    let mut grid = fusion_collide::Grid::new();
    let mut got = Vec::new();
    grid.find_pairs(&aabbs, &dynamic, &mut got);

    let mut deduped = got.clone();
    deduped.dedup();
    assert_eq!(got, deduped, "a pair was reported more than once");
}

/// A scene that is mostly enormous bodies has no cell size that fits, and the
/// pipeline is expected to use the sweep for it.
#[test]
fn the_grid_declines_scenes_it_does_not_suit() {
    let small = random_boxes(4_000, 3, 40.0);
    assert!(
        fusion_collide::Grid::suits(&small),
        "four thousand similar bodies is what the grid is for"
    );

    let mut lumpy = small.clone();
    for k in 0..1_000 {
        let s = 200.0 + k as f32;
        lumpy.push(Aabb::new(vec3(-s, -s, -s), vec3(s, s, s)));
    }
    assert!(
        !fusion_collide::Grid::suits(&lumpy),
        "a quarter of the scene being enormous is the heterogeneous case"
    );

    assert!(
        !fusion_collide::Grid::suits(&random_boxes(64, 3, 10.0)),
        "the sweep is already fast at sixty-four bodies"
    );
}

/// Scenes chosen to break a uniform grid.
fn grid_scenes() -> Vec<(&'static str, Vec<Aabb>)> {
    let mut with_floor = random_boxes(3_000, 7, 30.0);
    with_floor.push(Aabb::new(
        vec3(-400.0, -2.0, -400.0),
        vec3(400.0, 0.0, 400.0),
    ));

    let mut mixed = random_boxes(2_500, 9, 20.0);
    for k in 0..40 {
        let s = 2.0 + k as f32 * 0.5;
        let c = vec3(k as f32, 0.0, -(k as f32));
        mixed.push(Aabb::new(c - Vec3::splat(s), c + Vec3::splat(s)));
    }

    let far: Vec<Aabb> = random_boxes(2_200, 13, 25.0)
        .into_iter()
        .map(|b| {
            let shift = vec3(12_000.0, -3_000.0, 8_000.0);
            Aabb::new(b.min + shift, b.max + shift)
        })
        .collect();

    vec![
        ("uniform cloud", random_boxes(3_000, 5, 30.0)),
        ("dense cloud", random_boxes(3_000, 5, 8.0)),
        ("cloud and a floor", with_floor),
        ("mixed sizes", mixed),
        ("far from the origin", far),
    ]
}
