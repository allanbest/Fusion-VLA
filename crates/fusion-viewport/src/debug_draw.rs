//! Turning simulation state into debug geometry.
//!
//! PLAN.md §19: every visualization mode is the same geometry with a different
//! shading function, composed as overlays rather than selected with a radio
//! button. [`DrawStyle`] is the v0.1 version of that — a set of independent
//! toggles, not a mode enum.

use fusion_core::{BodySet, DistanceConstraint, ParticleSet, SurfaceTriangle};
use fusion_math::Vec3;

use crate::mesh::{Geometry, Surface};

/// How bodies are tinted apart from one another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Palette {
    /// Stepped hues. Individual motion is readable in a crowd, which is what
    /// a debug view of a thousand falling boxes needs.
    #[default]
    Hue,
    /// Stepped lightness, no hue at all. What the editor uses: its chrome is
    /// achromatic on purpose, and a viewport full of arbitrary colour would
    /// out-shout both the syntax-coloured code view and the one thing in the
    /// scene whose colour carries meaning — the red/green/blue gizmo.
    Neutral,
}

#[derive(Clone, Copy, Debug)]
pub struct DrawStyle {
    pub solid: bool,
    pub wireframe: bool,
    /// Broad-phase bounds, the first thing to look at when pairs go missing.
    pub aabbs: bool,
    /// Velocity arrows, scaled by `velocity_scale` seconds of travel.
    pub velocities: bool,
    pub velocity_scale: f32,
    pub grid: bool,
    /// How bodies are told apart from one another.
    pub palette: Palette,
}

impl Default for DrawStyle {
    fn default() -> Self {
        Self {
            solid: true,
            wireframe: false,
            aabbs: false,
            velocities: false,
            velocity_scale: 0.25,
            grid: true,
            palette: Palette::Hue,
        }
    }
}

/// Static bodies are the scenery; dynamic bodies get colour so individual
/// motion is readable in a crowd.
pub fn body_color(index: usize, dynamic: bool) -> [f32; 3] {
    body_color_in(Palette::Hue, index, dynamic)
}

/// The same, in a chosen palette.
///
/// Both palettes step by the golden ratio, so adjacent bodies never land on
/// neighbouring values however many there are. What differs is the axis they
/// step along: [`Palette::Hue`] walks the colour wheel, [`Palette::Neutral`]
/// walks lightness. Neutral gives up some separability — greys collide sooner
/// than hues do — and buys a viewport that does not fight an achromatic editor
/// for attention.
pub fn body_color_in(palette: Palette, index: usize, dynamic: bool) -> [f32; 3] {
    let step = (index as f32 * 0.618_034) % 1.0;
    match palette {
        Palette::Hue => {
            if !dynamic {
                return [0.30, 0.32, 0.36];
            }
            // Golden-ratio hue stepping: adjacent bodies never land on similar
            // hues.
            hsv_to_rgb(step, 0.55, 0.95)
        }
        Palette::Neutral => {
            if !dynamic {
                // Darker than any dynamic body, so the floor and the walls
                // read as the room rather than as more of the contents.
                return [0.20, 0.20, 0.22];
            }
            // A mid band, not the whole ramp. The editor outlines the
            // selection in near-white and paints its chrome in near-black, so
            // both ends of the ramp are spoken for: a body up at 0.9 would be
            // indistinguishable from its own selection outline.
            let value = 0.28 + step * 0.34;
            [value, value, value * 1.03]
        }
    }
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match (i as i32) % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Draw every body in a set.
pub fn bodies(bodies: &BodySet, style: &DrawStyle, g: &mut Geometry) {
    bodies_with_appearance(bodies, style, g, |_| None);
}

/// Draw bodies with authored colour, roughness and metallic values. Debug
/// overlays remain independent of appearance, and missing entries use the
/// chosen debug palette.
pub fn bodies_with_appearance(
    bodies: &BodySet,
    style: &DrawStyle,
    g: &mut Geometry,
    appearance: impl Fn(usize) -> Option<([f32; 3], Surface)>,
) {
    for i in 0..bodies.len() {
        if bodies.is_removed(i) {
            continue;
        }
        let pose = bodies.pose(i);
        let dynamic = bodies.is_dynamic(i);
        let (color, surface) = appearance(i).unwrap_or_else(|| {
            (
                body_color_in(style.palette, i, dynamic),
                surface_for(&bodies.material[i]),
            )
        });

        if style.solid {
            g.with_surface(surface, |g| {
                g.solid_shape(&bodies.shape[i], &pose, color);
            });
        }
        if style.wireframe || !style.solid {
            let wire = if style.solid {
                [color[0] * 0.35, color[1] * 0.35, color[2] * 0.35]
            } else {
                color
            };
            g.wire_shape(&bodies.shape[i], &pose, wire);
        }
        if style.aabbs {
            let b = bodies.aabb(i);
            g.wire_aabb(b.min, b.max, [0.25, 0.75, 0.35]);
        }
        if style.velocities && dynamic {
            let v = bodies.linear_velocity[i];
            if v.length_squared() > 1e-4 {
                g.arrow(
                    pose.translation,
                    pose.translation + v * style.velocity_scale,
                    [1.0, 0.9, 0.2],
                );
            }
        }
    }
}

/// [`bodies_with_appearance`], with a field painted over the bodies that
/// carry one: `field` gives, for a body, how finely to cut its faces and the
/// colour at each body-local point on its surface; a body it says nothing
/// for is drawn in its flat appearance colour.
pub fn bodies_with_field<F: Fn(Vec3) -> [f32; 3]>(
    bodies: &BodySet,
    style: &DrawStyle,
    g: &mut Geometry,
    appearance: impl Fn(usize) -> Option<([f32; 3], Surface)>,
    field: impl Fn(usize) -> Option<(f32, F)>,
) {
    if style.solid {
        for i in 0..bodies.len() {
            if bodies.is_removed(i) {
                continue;
            }
            let Some((detail, colour_at)) = field(i) else {
                continue;
            };
            let surface = appearance(i)
                .map(|(_, s)| s)
                .unwrap_or_else(|| surface_for(&bodies.material[i]));
            let pose = bodies.pose(i);
            g.with_surface(surface, |g| {
                g.solid_shape_field(&bodies.shape[i], &pose, detail, colour_at);
            });
        }
    }
    // Everything else — the bodies with no field, and every body's wires,
    // bounds and arrows — as usual, with the field bodies' solids skipped.
    let painted: Vec<bool> = (0..bodies.len())
        .map(|i| style.solid && field(i).is_some())
        .collect();
    for (i, &painted) in painted.iter().enumerate() {
        if bodies.is_removed(i) {
            continue;
        }
        let pose = bodies.pose(i);
        let dynamic = bodies.is_dynamic(i);
        let (color, surface) = appearance(i).unwrap_or_else(|| {
            (
                body_color_in(style.palette, i, dynamic),
                surface_for(&bodies.material[i]),
            )
        });
        if style.solid && !painted {
            g.with_surface(surface, |g| {
                g.solid_shape(&bodies.shape[i], &pose, color);
            });
        }
        if style.wireframe || !style.solid {
            let wire = if style.solid {
                [color[0] * 0.35, color[1] * 0.35, color[2] * 0.35]
            } else {
                color
            };
            g.wire_shape(&bodies.shape[i], &pose, wire);
        }
        if style.aabbs {
            let b = bodies.aabb(i);
            g.wire_aabb(b.min, b.max, [0.25, 0.75, 0.35]);
        }
        if style.velocities && dynamic {
            let v = bodies.linear_velocity[i];
            if v.length_squared() > 1e-4 {
                g.arrow(
                    pose.translation,
                    pose.translation + v * style.velocity_scale,
                    [1.0, 0.9, 0.2],
                );
            }
        }
    }
}

/// Draw a cloth surface.
///
/// Every triangle is emitted twice, once in each winding. Cloth has no inside,
/// and the renderer culls back faces — a single winding would make the sheet
/// vanish whenever it turned away from the camera, which for something that
/// flaps is most of the time.
pub fn cloth(
    particles: &ParticleSet,
    triangles: &[SurfaceTriangle],
    color: [f32; 3],
    wireframe: bool,
    g: &mut Geometry,
) {
    for t in triangles {
        let (a, b, c) = (
            particles.translation(t.a as usize),
            particles.translation(t.b as usize),
            particles.translation(t.c as usize),
        );
        if wireframe {
            let wire = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6];
            g.line(a, b, wire);
            g.line(b, c, wire);
            g.line(c, a, wire);
        } else {
            g.triangle_shaded(a, b, c, color);
            // Reverse winding, slightly darker so the far side of a fold reads
            // as a far side rather than as a lighting glitch.
            g.triangle_shaded(a, c, b, [color[0] * 0.7, color[1] * 0.7, color[2] * 0.8]);
        }
    }
}

/// Draw a particle lattice as its constraint edges.
///
/// Cloth has a triangle list and is drawn as a surface; a soft body is a
/// volume of particles with no surface mesh, so its edges are the only thing
/// there is to look at. Extracting a surface from a lattice is a
/// marching-cubes problem, and it belongs with the fluid surfacing scheduled
/// for v0.4 rather than here.
///
/// `first_lattice_particle` is where the caller's cloth ends and its
/// deformable solids begin — constraints entirely below it are already drawn
/// as a surface and are skipped.
pub fn particle_links(
    particles: &ParticleSet,
    constraints: &[DistanceConstraint],
    first_lattice_particle: usize,
    color: [f32; 3],
    g: &mut Geometry,
) {
    for c in constraints {
        let (a, b) = (c.a as usize, c.b as usize);
        if a < first_lattice_particle && b < first_lattice_particle {
            continue;
        }
        g.line(particles.translation(a), particles.translation(b), color);
    }
}

/// Pinned particles, so it is obvious what is holding a sheet up.
pub fn pinned_particles(particles: &ParticleSet, g: &mut Geometry) {
    for i in 0..particles.len() {
        if !particles.is_movable(i) {
            g.point(particles.translation(i), 0.03, [1.0, 0.85, 0.2]);
        }
    }
}

/// Contact points and their normals. Callers pass these in because contacts
/// live in `fusion-collide`, which the viewport deliberately does not depend
/// on — the dependency only ever points from applications down into the core.
pub fn contacts(points: &[(Vec3, Vec3, f32)], g: &mut Geometry) {
    for &(point, normal, depth) in points {
        // Red for penetrating, cyan for merely touching.
        let color = if depth < 0.0 {
            [1.0, 0.25, 0.25]
        } else {
            [0.2, 0.9, 0.95]
        };
        g.point(point, 0.05, color);
        g.line(point, point + normal * 0.35, color);
    }
}

/// How a physics material should look.
///
/// PLAN.md §12 makes a material "a constitutive law plus parameters", and the
/// parameters it carries are the ones the *solver* needs — density, friction,
/// restitution. None of those is a shading parameter, and inferring one from
/// the others would be inventing physics: a dense thing is not necessarily
/// metallic, and a bouncy one is not necessarily glossy.
///
/// So this is a lookup, not a derivation, and it is deliberately crude. It
/// exists so that a scene of steel and wood does not render as two identically
/// matte greys, which is a legibility problem rather than a physical claim.
/// A real material system with appearance data alongside the mechanics is a
/// v0.6 concern; when it arrives, this function goes.
fn surface_for(material: &fusion_core::Material) -> Surface {
    // Matched on density because that is the only field that separates the
    // library entries, and matched loosely because the point is a visual
    // family rather than an identification.
    let d = material.density;
    if d > 6000.0 {
        // Steel and the other heavy metals.
        Surface::METAL
    } else if d > 2000.0 {
        // Concrete, stone: rough, and rougher than the default.
        Surface {
            roughness: 0.92,
            metallic: 0.0,
        }
    } else if material.restitution > 0.6 {
        // Rubber: a dielectric with a visible sheen.
        Surface {
            roughness: 0.45,
            metallic: 0.0,
        }
    } else if d < 500.0 {
        // Cork, foam, light woods.
        Surface {
            roughness: 0.85,
            metallic: 0.0,
        }
    } else {
        Surface::MATTE
    }
}
