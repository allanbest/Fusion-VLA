//! The uniform grid, against brute force.
//!
//! A neighbour structure that misses a neighbour fails silently and looks like
//! a solver bug — the same argument that keeps `find_pairs_brute_force` around
//! for the broad phase. It has a second failure mode the broad phase does not:
//! returning the *same* neighbour twice, which no distance test catches and
//! which an SPH density sum reads as extra mass.

use fusion_collide::SpatialHash;
use fusion_math::DVec3;

/// A deterministic scatter of points. Not `rand` — a test that cannot be
/// reproduced from its own source is not much of a regression test.
fn scatter(n: usize, extent: f64) -> Vec<DVec3> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) * extent - extent * 0.5
    };
    (0..n).map(|_| DVec3::new(next(), next(), next())).collect()
}

fn brute_force(points: &[DVec3], p: DVec3, radius: f32) -> Vec<u32> {
    let r_sq = (radius as f64) * (radius as f64);
    (0..points.len() as u32)
        .filter(|&i| (points[i as usize] - p).length_squared() <= r_sq)
        .collect()
}

#[test]
fn neighbours_match_brute_force() {
    let points = scatter(2000, 4.0);
    let radius = 0.3;
    let mut grid = SpatialHash::new();
    grid.build(&points, radius);

    let mut out = Vec::new();
    for &p in points.iter().take(200) {
        grid.neighbours(&points, p, radius, &mut out);
        assert_eq!(out, brute_force(&points, p, radius));
    }
}

#[test]
fn a_query_visits_each_point_at_most_once() {
    // The failure this guards is a hash collision between two of the 27 cells
    // a query walks, which enumerates one bucket twice. With 27 draws from a
    // small table it happens often enough to be the common case, not the rare
    // one, and it inflated an SPH density estimate by 9% on a lattice that had
    // not moved.
    let points = scatter(1500, 3.0);
    let radius = 0.25;
    let mut grid = SpatialHash::new();
    grid.build(&points, radius);

    for &p in points.iter().take(300) {
        let mut seen = Vec::new();
        grid.for_each_candidate(p, |i| seen.push(i));
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(before, seen.len(), "a candidate was reported twice");
    }
}

#[test]
fn a_far_away_scene_gets_the_same_neighbours() {
    // Cell coordinates are taken from a snapped local origin, not from the
    // world origin, so a scene 20 km out is indexed exactly as one at the
    // origin. Without that, the integer cell coordinate overflows into the
    // hash's high bits and the bucket distribution goes with it.
    let near = scatter(800, 2.0);
    let shift = DVec3::new(20_000.0, -5_000.0, 12_000.0);
    let far: Vec<DVec3> = near.iter().map(|p| *p + shift).collect();
    let radius = 0.2;

    let (mut a, mut b) = (SpatialHash::new(), SpatialHash::new());
    a.build(&near, radius);
    b.build(&far, radius);

    let (mut out_a, mut out_b) = (Vec::new(), Vec::new());
    for i in 0..200 {
        a.neighbours(&near, near[i], radius, &mut out_a);
        b.neighbours(&far, far[i], radius, &mut out_b);
        assert_eq!(out_a, out_b, "neighbour sets diverged at point {i}");
    }
}

#[test]
fn rebuilding_the_same_points_gives_the_same_order() {
    // Determinism (PLAN.md §34). Floating-point addition is not associative,
    // so a neighbour list in a different order is a different density, and a
    // different density is a different simulation.
    let points = scatter(1000, 2.0);
    let radius = 0.25;

    let mut first = SpatialHash::new();
    first.build(&points, radius);
    let mut second = SpatialHash::new();
    // Build something else in between, so any reuse of internal state shows up.
    second.build(&scatter(37, 9.0), 1.0);
    second.build(&points, radius);

    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &p in &points {
        a.clear();
        b.clear();
        first.for_each_candidate(p, |i| a.push(i));
        second.for_each_candidate(p, |i| b.push(i));
        assert_eq!(a, b);
    }
}

#[test]
fn an_empty_or_single_point_grid_is_harmless() {
    let mut grid = SpatialHash::new();
    grid.build(&[], 0.1);
    assert!(grid.is_empty());
    grid.for_each_candidate(DVec3::ZERO, |_| panic!("empty grid yielded a point"));

    let one = [DVec3::new(1.0, 2.0, 3.0)];
    grid.build(&one, 0.1);
    let mut out = Vec::new();
    grid.neighbours(&one, one[0], 0.1, &mut out);
    assert_eq!(out, vec![0]);
}

/// The cell-major query has to see exactly the points the per-point query
/// sees, in exactly the same order.
///
/// This is the invariant the fluid's neighbour build rests on: it stopped
/// asking "what is near this point" and started asking "what is near this
/// cell", once per cell instead of once per particle, on the assumption that
/// the two answers are the same set in the same sequence. If they ever differ,
/// the neighbour lists change, and with them every fluid result and the
/// bit-identical replay guarantee in PLAN.md §34.
#[test]
fn candidate_spans_match_the_per_point_walk() {
    let mut points = Vec::new();
    let mut seed = 0x5eed_u32;
    let mut random = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f64 / 16_777_216.0
    };
    for _ in 0..4000 {
        points.push(DVec3::new(
            random() * 3.0 - 1.5,
            random() * 3.0 - 1.5,
            random() * 3.0 - 1.5,
        ));
    }

    let mut grid = SpatialHash::new();
    grid.build(&points, 0.25);
    let (xs, ys, zs) = grid.sorted_axes();
    let ids = grid.sorted_indices();
    let mut spans = Vec::new();

    for k in 0..points.len() {
        let mut expected = Vec::new();
        grid.for_each_candidate(points[ids[k] as usize], |i| expected.push(i));

        let local = fusion_math::Vec3::new(xs[k], ys[k], zs[k]);
        grid.candidate_spans(grid.cell_of_local(local), &mut spans);
        let mut found = Vec::new();
        for &(from, to) in &spans {
            found.extend_from_slice(&ids[from as usize..to as usize]);
        }
        assert_eq!(found, expected, "point {k}");
    }
}

/// The axis arrays are the same points as the index array, permuted the same
/// way and moved into the grid's local frame.
#[test]
fn sorted_axes_agree_with_the_points_they_index() {
    let points: Vec<DVec3> = (0..500)
        .map(|i| {
            let t = i as f64 * 0.37;
            // Offset a long way from the origin: the local frame is what makes
            // single precision safe here, and a test at the origin would not
            // notice if it were dropped.
            DVec3::new(
                2_000.0 + t.sin(),
                2_000.0 + t.cos() * 2.0,
                2_000.0 + t * 0.01,
            )
        })
        .collect();
    let mut grid = SpatialHash::new();
    grid.build(&points, 0.3);

    let (xs, ys, zs) = grid.sorted_axes();
    let ids = grid.sorted_indices();
    assert_eq!(xs.len(), ids.len());
    for k in 0..ids.len() {
        let expected = points[ids[k] as usize] - grid.origin();
        for (got, want) in [
            (xs[k], expected.x),
            (ys[k], expected.y),
            (zs[k], expected.z),
        ] {
            // A metre of local extent in f32 is resolved to well under a
            // micron; the tolerance is what the narrowing costs, not slack.
            assert!(
                (got as f64 - want).abs() < 1e-6,
                "axis {got} vs {want} at slot {k}"
            );
        }
    }
}

/// Every point is its own neighbour — including points sitting exactly on a
/// cell boundary.
///
/// The sharp form of the invariant that the build and the query must agree on
/// which cell a point is in. If they ever disagree, the point is filed under
/// one cell and looked for around another, and the 27-cell walk comes back
/// centred one cell over: a whole face of genuine neighbours goes missing and
/// the point cannot even find itself. Boundary coordinates are the adversarial
/// case, because that is where a narrowing conversion changes the answer.
#[test]
fn a_point_on_a_cell_boundary_finds_itself() {
    let cell = 0.25f64;
    let mut points = Vec::new();
    for i in 0..20 {
        for j in 0..20 {
            // Exact multiples of the cell size, and a hair either side of one.
            let (x, y) = (i as f64 * cell, j as f64 * cell);
            points.push(DVec3::new(x, y, 0.0));
            points.push(DVec3::new(x - 1e-15, y, cell));
            points.push(DVec3::new(x + 1e-15, y, 2.0 * cell));
        }
    }
    // Repeat the whole thing a long way out, where the narrowing to the local
    // frame actually costs something.
    let far = DVec3::new(3_000.0, -1_500.0, 800.0);
    points.extend_from_slice(&points.clone().iter().map(|p| *p + far).collect::<Vec<_>>());

    let mut grid = SpatialHash::new();
    grid.build(&points, cell as f32);
    for (i, p) in points.iter().enumerate() {
        let mut found = false;
        grid.for_each_candidate(*p, |j| found |= j as usize == i);
        assert!(found, "point {i} at {p:?} is not in its own candidate set");
    }
}
