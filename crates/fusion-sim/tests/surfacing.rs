//! Marching cubes, against analytic surfaces.
//!
//! The polygonizer is driven by a 256-row table transcribed by hand, and a
//! single wrong entry produces a mesh that looks fine and is not. These tests
//! are chosen so that a wrong entry cannot survive them: a hole breaks
//! `is_closed`, a mirrored triple breaks the sign of the enclosed volume, and
//! a missing triple leaves both the volume and the area short by an amount far
//! larger than the discretization error.

use fusion_core::{FluidDesc, FluidParams, World};
use fusion_math::{vec3, Aabb, Vec3};
use fusion_sim::surface::{SurfaceMesh, SurfaceSettings, Surfacer};
use fusion_sim::Simulation;

const MAX_VERTICES: usize = 24_000_000;

fn sphere_field(centre: Vec3, radius: f32) -> impl Fn(Vec3) -> f32 {
    move |p| (p - centre).length() - radius
}

#[test]
fn a_sphere_comes_out_closed_and_the_right_size() {
    let radius = 1.0f32;
    let centre = vec3(0.13, -0.07, 0.21); // Off-lattice on purpose.
    let bounds = Aabb::from_center_half_extents(centre, Vec3::splat(radius * 1.4));
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();

    surfacer.extract_field(
        &bounds,
        0.05,
        MAX_VERTICES,
        sphere_field(centre, radius),
        &mut mesh,
    );

    assert!(!mesh.is_empty());
    assert!(mesh.is_closed(), "the mesh has a hole or a duplicated face");

    let expected_volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    let expected_area = 4.0 * std::f32::consts::PI * radius * radius;
    // The volume is computed about the origin, and the sphere is not at the
    // origin — which is the point. The divergence-theorem sum is only
    // translation-invariant for a closed mesh, so a hole shows up here as a
    // wildly wrong number rather than a slightly wrong one.
    let volume = mesh.volume();
    let area = mesh.area();
    assert!(
        (volume - expected_volume).abs() / expected_volume < 0.01,
        "volume {volume:.4} m^3 against an analytic {expected_volume:.4}"
    );
    assert!(
        (area - expected_area).abs() / expected_area < 0.02,
        "area {area:.4} m^2 against an analytic {expected_area:.4}"
    );
}

#[test]
fn normals_point_out_of_the_surface() {
    // Winding, and the whole reason the triangle table is read backwards: get
    // it wrong and every surface in the engine renders inside out.
    let radius = 1.0f32;
    let bounds = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.4));
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    surfacer.extract_field(
        &bounds,
        0.06,
        MAX_VERTICES,
        sphere_field(Vec3::ZERO, radius),
        &mut mesh,
    );

    assert!(mesh.volume() > 0.0, "winding is inverted");
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        assert!(
            n.dot(p.normalize_or_zero()) > 0.5,
            "normal at {p:?} points inward"
        );
    }
}

#[test]
fn refining_the_lattice_converges_on_the_analytic_answer() {
    // The property that separates a correct polygonizer from one that merely
    // produces triangles: the error has to go down when the cells get smaller.
    let radius = 1.0f32;
    let exact = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    let bounds = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.3));
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();

    let mut previous = f32::INFINITY;
    for cell in [0.2f32, 0.1, 0.05, 0.025] {
        surfacer.extract_field(
            &bounds,
            cell,
            MAX_VERTICES,
            sphere_field(Vec3::ZERO, radius),
            &mut mesh,
        );
        let error = (mesh.volume() - exact).abs() / exact;
        assert!(
            error < previous,
            "error rose from {previous:.5} to {error:.5} at cell size {cell}"
        );
        previous = error;
    }
    assert!(previous < 0.002, "finest error is still {previous:.5}");
}

#[test]
fn a_torus_is_closed_too() {
    // Genus one. A sphere can be polygonized correctly by a table that is
    // wrong about the configurations where a cell is cut twice; a torus
    // produces those cells in quantity.
    let (major, minor) = (1.0f32, 0.35f32);
    let field = move |p: Vec3| {
        let radial = (p.x * p.x + p.z * p.z).sqrt() - major;
        (radial * radial + p.y * p.y).sqrt() - minor
    };
    let bounds = Aabb::from_center_half_extents(
        Vec3::ZERO,
        vec3(major + minor + 0.2, minor + 0.2, major + minor + 0.2),
    );
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    surfacer.extract_field(&bounds, 0.03, MAX_VERTICES, field, &mut mesh);

    assert!(mesh.is_closed(), "the torus has a hole");
    let exact = 2.0 * std::f32::consts::PI.powi(2) * major * minor * minor;
    let volume = mesh.volume();
    assert!(
        (volume - exact).abs() / exact < 0.01,
        "torus volume {volume:.4} against an analytic {exact:.4}"
    );
}

#[test]
fn two_separated_blobs_are_two_closed_surfaces() {
    let field = move |p: Vec3| {
        let a = (p - vec3(-0.7, 0.0, 0.0)).length() - 0.4;
        let b = (p - vec3(0.7, 0.0, 0.0)).length() - 0.25;
        a.min(b)
    };
    let bounds = Aabb::from_center_half_extents(Vec3::ZERO, vec3(1.4, 0.7, 0.7));
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    surfacer.extract_field(&bounds, 0.025, MAX_VERTICES, field, &mut mesh);

    assert!(mesh.is_closed());
    let exact = (4.0 / 3.0) * std::f32::consts::PI * (0.4f32.powi(3) + 0.25f32.powi(3));
    assert!((mesh.volume() - exact).abs() / exact < 0.02);
}

#[test]
fn an_entirely_inside_or_entirely_outside_field_produces_nothing() {
    let bounds = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::ONE);
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();

    surfacer.extract_field(&bounds, 0.1, MAX_VERTICES, |_| 1.0, &mut mesh);
    assert!(mesh.is_empty());

    surfacer.extract_field(&bounds, 0.1, MAX_VERTICES, |_| -1.0, &mut mesh);
    // The field never changes sign anywhere in the lattice, including at its
    // edge, so there is no surface to find. A polygonizer that closed the
    // volume against the boundary of its own grid would produce a box here,
    // and that box would be an artefact of the grid rather than of the data.
    assert!(mesh.is_empty());
}

#[test]
fn the_lattice_is_coarsened_rather_than_allocating_without_limit() {
    // A 20 m box at a 1 mm cell size is 8 x 10^12 vertices. The request is
    // honoured as far as it can be and reported honestly, which is the
    // difference between a slow frame and a dead process.
    let bounds = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(10.0));
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    surfacer.extract_field(
        &bounds,
        0.001,
        200_000,
        sphere_field(Vec3::ZERO, 5.0),
        &mut mesh,
    );

    let dims = surfacer.last_dimensions();
    assert!(dims[0] * dims[1] * dims[2] <= 200_000);
    assert!(
        surfacer.last_cell_size() > 0.001,
        "cell size was not coarsened"
    );
    assert!(mesh.is_closed());
}

#[test]
fn surfacing_a_settled_fluid_recovers_its_volume() {
    // The end-to-end claim: the mesh a renderer would draw encloses the amount
    // of water the simulation says it has. Everything upstream has to be right
    // for this to hold — the fill, the density calibration, the boundary
    // correction, and the surface radius that decides where "the surface" is
    // relative to the outermost particle centres.
    let mut world = World::new();
    let t = 0.2;
    world.add(
        fusion_core::BodyDesc::cuboid(vec3(1.4, t, 1.4))
            .at(vec3(0.0, -t * 0.5, 0.0))
            .fixed(),
    );
    for (dx, dz) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let size = if dx != 0.0 {
            vec3(t, 1.5, 1.0)
        } else {
            vec3(1.0, 1.5, t)
        };
        world.add(
            fusion_core::BodyDesc::cuboid(size)
                .at(vec3(dx * 0.6, 0.75, dz * 0.6))
                .fixed(),
        );
    }
    let params = FluidParams::water_at(0.05);
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.8))
            .at(vec3(0.0, 0.65, 0.0))
            .with_params(params),
    );
    let expected = world.fluid(fluid).mass() / params.rest_density;

    let mut sim = Simulation::new(world);
    sim.run_frames(150, |_, _| {});

    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    let settings = SurfaceSettings::for_fluid(sim.world.fluid(fluid));
    surfacer.extract_fluid(sim.world.fluid(fluid), &settings, &mut mesh);

    assert!(!mesh.is_empty(), "settled water produced no surface");
    assert!(mesh.is_closed(), "the water surface is not closed");
    let volume = mesh.volume();
    assert!(
        (volume - expected).abs() / expected < 0.12,
        "surface encloses {volume:.4} m^3, the fluid holds {expected:.4} m^3"
    );
}

#[test]
fn surfacing_the_same_particles_twice_gives_the_same_mesh() {
    // PLAN.md §34 again: a mesh that differs between runs makes a baked
    // sequence non-reproducible even when the simulation behind it is not.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    let fluid = world
        .add_fluid(&FluidDesc::block(Vec3::splat(0.4)).with_params(FluidParams::water_at(0.05)));
    let settings = SurfaceSettings::for_fluid(world.fluid(fluid));

    let mut a = SurfaceMesh::default();
    let mut b = SurfaceMesh::default();
    let mut surfacer = Surfacer::new();
    surfacer.extract_fluid(world.fluid(fluid), &settings, &mut a);
    // Reuse the same surfacer, so any leftover state from the first pass would
    // show up in the second.
    surfacer.extract_fluid(world.fluid(fluid), &settings, &mut b);
    assert_eq!(a, b);

    let mut fresh = Surfacer::new();
    let mut c = SurfaceMesh::default();
    fresh.extract_fluid(world.fluid(fluid), &settings, &mut c);
    assert_eq!(a, c);
}

#[test]
fn the_ambiguous_face_problem_is_where_it_is_expected_and_nowhere_else() {
    // The classic marching-cubes table is not face-consistent: two cells can
    // disagree about how they cut the face they share, leaving a gap. This
    // test does not assert the gap away — it pins down where it lives, so that
    // "the surface has a hole" is a measurement rather than a surprise.
    //
    // A field whose sign flips between every adjacent sample is the worst case
    // and is not a thing anyone surfaces: it means the lattice is far coarser
    // than the feature. A field sampled finely enough to be worth
    // polygonizing produces no ambiguous faces at all.
    let n = 32usize;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    };

    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();

    let random: Vec<f32> = (0..n * n * n).map(|_| next()).collect();
    surfacer.polygonize_field([n; 3], 1.0, Vec3::ZERO, &random, &mut mesh);
    assert!(
        !mesh.is_closed(),
        "the adversarial field stopped producing ambiguous faces — if the \
         table gained the asymptotic decider, delete this half of the test"
    );
    let bad = non_manifold_fraction(&mesh);
    assert!(
        bad < 0.06,
        "non-manifold edges rose to {:.2}% on the random field",
        bad * 100.0
    );

    let mut smooth = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = vec3(x as f32, y as f32, z as f32) / n as f32 * 2.0 - Vec3::ONE;
                smooth[(z * n + y) * n + x] = p.length() - 0.7 + next() * 0.02;
            }
        }
    }
    surfacer.polygonize_field([n; 3], 1.0, Vec3::ZERO, &smooth, &mut mesh);
    assert!(
        mesh.is_closed(),
        "a sphere with 2% noise produced {:.2}% non-manifold edges",
        non_manifold_fraction(&mesh) * 100.0
    );
}

/// Fraction of edges not shared by exactly two triangles.
fn non_manifold_fraction(mesh: &SurfaceMesh) -> f32 {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for t in mesh.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    edges.sort_unstable();
    let (mut bad, mut total) = (0usize, 0usize);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        total += 1;
        if j - i != 2 {
            bad += 1;
        }
        i = j;
    }
    bad as f32 / total.max(1) as f32
}

#[test]
fn an_empty_point_cloud_surfaces_to_nothing() {
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    let settings = SurfaceSettings {
        cell_size: 0.05,
        influence: 0.1,
        radius: 0.025,
        padding: 0.1,
        max_vertices: MAX_VERTICES,
    };
    surfacer.extract_points(&[], &settings, &mut mesh);
    assert!(mesh.is_empty());
    assert!(mesh.positions.is_empty());
}

#[test]
fn a_single_particle_surfaces_to_a_small_closed_blob() {
    let mut surfacer = Surfacer::new();
    let mut mesh = SurfaceMesh::default();
    let settings = SurfaceSettings {
        cell_size: 0.01,
        influence: 0.1,
        radius: 0.05,
        padding: 0.1,
        max_vertices: MAX_VERTICES,
    };
    surfacer.extract_points(&[vec3(0.3, -0.2, 0.1)], &settings, &mut mesh);
    assert!(mesh.is_closed());
    // One particle gives exactly a sphere of the surface radius: the weighted
    // average of one point is that point.
    let exact = (4.0 / 3.0) * std::f32::consts::PI * 0.05f32.powi(3);
    assert!(
        (mesh.volume() - exact).abs() / exact < 0.05,
        "single-particle blob encloses {:.6} m^3, a sphere of r = 0.05 is {exact:.6}",
        mesh.volume()
    );
}
