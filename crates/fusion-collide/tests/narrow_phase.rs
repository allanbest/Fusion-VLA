//! Narrow-phase tests against closed-form answers.
//!
//! PLAN.md §30 asks for GJK/EPA checked against analytic cases. Where an exact
//! answer exists, that is what these assert on — a collision routine that is
//! merely self-consistent is how an engine ships a wrong normal for a year.

use fusion_collide::contact::WorldContact;
use fusion_collide::narrow::{collide_gjk_epa, gjk, primitives};
use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Quat, Vec3};

fn at(p: Vec3) -> Isometry {
    Isometry::from_translation(p)
}

/// `Shape::cuboid` takes a full size; these fixtures are easier to reason
/// about in half-extents, so they say so explicitly.
fn boxx(half: Vec3) -> Shape {
    Shape::Cuboid { half_extents: half }
}

fn cube(half: f32) -> Shape {
    boxx(Vec3::splat(half))
}

fn single(contacts: &[WorldContact]) -> WorldContact {
    assert_eq!(
        contacts.len(),
        1,
        "expected exactly one contact, got {}",
        contacts.len()
    );
    contacts[0]
}

#[test]
fn sphere_sphere_matches_closed_form() {
    let mut out = Vec::new();
    // Radii 1 and 1, centres 1.5 apart: 0.5 of overlap along +x.
    primitives::sphere_sphere(Vec3::ZERO, 1.0, vec3(1.5, 0.0, 0.0), 1.0, 0.0, &mut out);
    let c = single(&out);
    assert!(
        c.normal.approx_eq(Vec3::X, 1e-5),
        "normal must run from A to B, got {:?}",
        c.normal
    );
    assert!(
        (c.separation + 0.5).abs() < 1e-5,
        "separation {}",
        c.separation
    );
    assert!(c.point_a.approx_eq(vec3(1.0, 0.0, 0.0), 1e-5));
    assert!(c.point_b.approx_eq(vec3(0.5, 0.0, 0.0), 1e-5));

    // Apart: no contact.
    out.clear();
    primitives::sphere_sphere(Vec3::ZERO, 1.0, vec3(2.5, 0.0, 0.0), 1.0, 0.0, &mut out);
    assert!(out.is_empty());
}

#[test]
fn gjk_epa_agrees_with_the_analytic_sphere_case() {
    let a = Shape::sphere(1.0);
    let b = Shape::sphere(0.75);
    let pose_a = at(Vec3::ZERO);
    let pose_b = at(vec3(1.0, 0.5, 0.25));

    let mut analytic = Vec::new();
    primitives::sphere_sphere(
        pose_a.translation,
        1.0,
        pose_b.translation,
        0.75,
        0.0,
        &mut analytic,
    );
    let mut general = Vec::new();
    collide_gjk_epa(&a, &pose_a, &b, &pose_b, &mut general);

    // EPA converges on a *polytope* approximation of a sphere, so it agrees
    // with the closed form to within its expansion tolerance and no further.
    // That gap is exactly why the analytic fast paths exist rather than
    // routing everything through the general algorithm.
    let (x, y) = (single(&analytic), single(&general));
    assert!(
        x.normal.approx_eq(y.normal, 2e-2),
        "{:?} vs {:?}",
        x.normal,
        y.normal
    );
    assert!(
        (x.separation - y.separation).abs() < 2e-2,
        "{} vs {}",
        x.separation,
        y.separation
    );
}

#[test]
fn gjk_epa_recovers_a_known_box_overlap() {
    // Half-extent-1 cubes, centres 1.6 apart along x: 0.4 of overlap, +x.
    let cube = cube(1.0);
    let mut out = Vec::new();
    collide_gjk_epa(
        &cube,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(1.6, 0.0, 0.0)),
        &mut out,
    );
    let c = single(&out);
    assert!(c.normal.approx_eq(Vec3::X, 1e-3), "got {:?}", c.normal);
    assert!((c.separation + 0.4).abs() < 1e-3, "got {}", c.separation);
}

#[test]
fn gjk_reports_the_exact_gap_between_separated_boxes() {
    let cube = cube(0.5);
    match gjk::intersect(&cube, &at(Vec3::ZERO), &cube, &at(vec3(3.0, 0.0, 0.0))) {
        gjk::GjkResult::Separated {
            distance,
            point_a,
            point_b,
        } => {
            assert!(
                (distance - 2.0).abs() < 1e-3,
                "gap should be 2.0, got {distance}"
            );
            assert!((point_a.x - 0.5).abs() < 1e-3, "{point_a:?}");
            assert!((point_b.x - 2.5).abs() < 1e-3, "{point_b:?}");
        }
        other => panic!("expected separation, got {other:?}"),
    }
}

#[test]
fn sphere_box_outside_and_inside() {
    let half = Vec3::splat(1.0);
    let mut out = Vec::new();

    // Sphere sitting above the +y face, overlapping by 0.25.
    primitives::sphere_cuboid(
        vec3(0.0, 1.75, 0.0),
        1.0,
        half,
        &at(Vec3::ZERO),
        0.0,
        &mut out,
    );
    let c = single(&out);
    assert!(
        c.normal.approx_eq(-Vec3::Y, 1e-5),
        "sphere -> box normal points down: {:?}",
        c.normal
    );
    assert!((c.separation + 0.25).abs() < 1e-5);
    assert!(c.point_b.approx_eq(vec3(0.0, 1.0, 0.0), 1e-5));

    // Sphere centre inside the box: escape through the nearest face (+x).
    out.clear();
    primitives::sphere_cuboid(
        vec3(0.9, 0.0, 0.0),
        0.2,
        half,
        &at(Vec3::ZERO),
        0.0,
        &mut out,
    );
    let c = single(&out);
    assert!(c.normal.approx_eq(-Vec3::X, 1e-5), "got {:?}", c.normal);
    assert!(
        (c.separation + (0.2 + 0.1)).abs() < 1e-5,
        "got {}",
        c.separation
    );

    // Diagonal approach against a corner.
    out.clear();
    let d = 1.0 + 0.5 / 3f32.sqrt();
    primitives::sphere_cuboid(vec3(d, d, d), 0.6, half, &at(Vec3::ZERO), 0.0, &mut out);
    let c = single(&out);
    assert!(
        c.normal.approx_eq(-Vec3::splat(1.0).normalize(), 1e-4),
        "got {:?}",
        c.normal
    );
}

#[test]
fn resting_box_produces_a_four_point_face_manifold() {
    // A stable resting contact needs a patch, not a point: PLAN.md §10 calls
    // stacking a persistence and iteration problem, but it starts here.
    let ground = boxx(vec3(10.0, 1.0, 10.0));
    let cube = cube(1.0);
    let mut out = Vec::new();
    // Ground top face at y = 1, cube half-height 1: resting centre is y = 2,
    // so y = 1.99 puts every corner 0.01 deep.
    primitives::collide(
        &ground,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(0.0, 1.99, 0.0)),
        0.0,
        &mut out,
    );
    assert_eq!(out.len(), 4, "expected a square patch, got {out:?}");
    for c in &out {
        assert!(
            c.normal.approx_eq(Vec3::Y, 1e-4),
            "normal points up from ground to box"
        );
        assert!(
            (c.separation + 0.01).abs() < 1e-3,
            "each corner is 0.01 deep, got {}",
            c.separation
        );
    }
    // The four points should span an area, not cluster.
    let spread = out
        .iter()
        .map(|c| c.point_b.x.abs().max(c.point_b.z.abs()))
        .fold(0.0f32, f32::max);
    assert!(spread > 0.9, "contact patch collapsed to a line or point");
}

#[test]
fn tilted_box_on_ground_contacts_one_edge() {
    let ground = boxx(vec3(10.0, 1.0, 10.0));
    let cube = cube(1.0);
    let tilt = Quat::from_rotation_z(0.35);
    // A cube tilted by 0.35 rad reaches down to 1 - cos(0.35) - sin(0.35)
    // below its centre; place it so only the low edge is inside the ground.
    let pose = Isometry::new(
        vec3(0.0, 1.0 + 1.0 * (0.35f32.cos() + 0.35f32.sin()) - 0.02, 0.0),
        tilt,
    );
    let mut out = Vec::new();
    primitives::collide(&ground, &at(Vec3::ZERO), &cube, &pose, 0.0, &mut out);
    assert!(
        !out.is_empty(),
        "a tilted box resting on the ground must contact it"
    );
    assert!(out.len() <= 4);
    for c in &out {
        assert!(c.normal.approx_eq(Vec3::Y, 1e-3), "got {:?}", c.normal);
        assert!(c.separation < 0.0);
    }
}

#[test]
fn separated_boxes_report_nothing() {
    let cube = cube(0.5);
    let mut out = Vec::new();
    primitives::collide(
        &cube,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(1.01, 0.0, 0.0)),
        0.0,
        &mut out,
    );
    assert!(
        out.is_empty(),
        "boxes 0.01 apart must not report contact: {out:?}"
    );
}

/// Boxes that are apart but *within* the tolerance report contact, with a
/// positive separation. This is what a speculative contact is.
///
/// It is the test that was missing, and what it cost is worth stating plainly:
/// `cuboid_cuboid` accepted a tolerance, used it only in the last few lines to
/// decide which clipped points to keep, and returned long before reaching them
/// — the separating-axis test gave up the instant the boxes stopped touching.
/// So **no pair of boxes had ever produced a speculative contact**, on any
/// version of Fusion. Spheres always had, which is why the symptom looked like
/// a friction problem: a thrown ball rolled and a thrown box, which fell 10 mm
/// through the floor before anything noticed, stopped dead where it landed.
#[test]
fn boxes_within_the_tolerance_report_a_gap() {
    let cube = cube(0.5);
    for gap in [0.001f32, 0.005, 0.019] {
        let mut out = Vec::new();
        primitives::collide(
            &cube,
            &at(Vec3::ZERO),
            &cube,
            &at(vec3(0.0, 1.0 + gap, 0.0)),
            0.02,
            &mut out,
        );
        assert!(
            !out.is_empty(),
            "a {gap} m gap inside a 0.02 m tolerance reported no contact"
        );
        for c in &out {
            assert!(
                (c.separation - gap).abs() < 1e-4,
                "separation {:.5} for a gap of {gap}",
                c.separation
            );
            assert!(c.normal.approx_eq(Vec3::Y, 1e-3), "got {:?}", c.normal);
        }
        // A face-on approach is a four-point patch, the same as a resting one.
        assert_eq!(out.len(), 4, "a face gap should give four points");
    }

    // And past the tolerance there is still nothing.
    let mut out = Vec::new();
    primitives::collide(
        &cube,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(0.0, 1.03, 0.0)),
        0.02,
        &mut out,
    );
    assert!(out.is_empty(), "a 0.03 m gap is outside a 0.02 m tolerance");
}

/// The edge-edge branch has to survive the same widening.
///
/// The bias that makes face axes win ties is a multiplication, and a
/// multiplication changes sign of meaning when the depth does: scaling a
/// *negative* depth by 1.02 makes it more negative, which would hand every
/// near-miss to an edge axis and give a flat landing a one-point manifold.
#[test]
fn a_tilted_box_within_the_tolerance_reports_an_edge() {
    let cube = cube(0.5);
    let tilted = Isometry::new(
        vec3(0.0, 0.5 + 0.5 * 2f32.sqrt() + 0.004, 0.0),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
    );
    let mut out = Vec::new();
    primitives::collide(&cube, &at(Vec3::ZERO), &cube, &tilted, 0.02, &mut out);
    assert!(!out.is_empty(), "a tilted box 4 mm away reported nothing");
    for c in &out {
        assert!(
            c.separation > 0.0 && c.separation < 0.02,
            "separation {:.5} outside the tolerance",
            c.separation
        );
    }
}

#[test]
fn box_box_normal_convention_survives_argument_order() {
    let cube = cube(0.5);
    let pose_a = at(Vec3::ZERO);
    let pose_b = at(vec3(0.0, 0.8, 0.0));

    let mut ab = Vec::new();
    primitives::collide(&cube, &pose_a, &cube, &pose_b, 0.0, &mut ab);
    let mut ba = Vec::new();
    primitives::collide(&cube, &pose_b, &cube, &pose_a, 0.0, &mut ba);

    assert!(!ab.is_empty() && !ba.is_empty());
    assert!(
        ab[0].normal.approx_eq(Vec3::Y, 1e-4),
        "A->B is +y, got {:?}",
        ab[0].normal
    );
    assert!(
        ba[0].normal.approx_eq(-Vec3::Y, 1e-4),
        "swapping the pair flips the normal"
    );
    assert!(
        (ab[0].separation - ba[0].separation).abs() < 1e-4,
        "depth must not depend on order"
    );
}

#[test]
fn edge_edge_contact_between_crossed_boxes() {
    // Two cubes each rolled 45 degrees, about different axes: the lower one
    // presents an upward edge running along x, the upper one a downward edge
    // running along z. Nothing is face-to-face, so the separating axis is a
    // cross product of edges and the manifold is a single point.
    let c = cube(0.5);
    let half_diagonal = 0.5 * 2f32.sqrt();
    let lower = Isometry::new(
        Vec3::ZERO,
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_4),
    );
    let upper = Isometry::new(
        vec3(0.0, 2.0 * half_diagonal - 0.1, 0.0),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
    );

    let mut out = Vec::new();
    primitives::collide(&c, &lower, &c, &upper, 0.0, &mut out);
    let contact = single(&out);
    assert!(
        contact.normal.approx_eq(Vec3::Y, 1e-3),
        "got {:?}",
        contact.normal
    );
    assert!(
        (contact.separation + 0.1).abs() < 1e-3,
        "got {}",
        contact.separation
    );
    // The edges cross above the origin.
    assert!(contact.point_a.x.abs() < 1e-3 && contact.point_a.z.abs() < 1e-3);
}

#[test]
fn sphere_box_flip_matches_box_sphere() {
    let sphere = Shape::sphere(0.5);
    let cube = cube(1.0);
    let ps = at(vec3(0.0, 1.3, 0.0));
    let pc = at(Vec3::ZERO);

    let mut sb = Vec::new();
    primitives::collide(&sphere, &ps, &cube, &pc, 0.0, &mut sb);
    let mut bs = Vec::new();
    primitives::collide(&cube, &pc, &sphere, &ps, 0.0, &mut bs);

    let (x, y) = (single(&sb), single(&bs));
    assert!(x.normal.approx_eq(-y.normal, 1e-5));
    assert!((x.separation - y.separation).abs() < 1e-6);
    assert!(x.point_a.approx_eq(y.point_b, 1e-5));
}
