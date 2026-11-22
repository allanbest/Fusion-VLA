//! Materials.
//!
//! PLAN.md §12: a material is a constitutive law plus parameters. v0.1 has no
//! constitutive laws, so this is only the parameter half, and only the handful
//! of parameters Basic reads. Friction and restitution are carried but unused
//! until v0.2 — they are here so that adding the friction solver does not
//! change every scene-construction call site.

use fusion_math::Vec3;

/// Bulk properties of a body. Densities are kg/m^3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Mass per unit volume, kg/m^3.
    pub density: f32,
    /// Coulomb friction coefficient. Unused in v0.1.
    pub friction: f32,
    /// Restitution in [0, 1]. Unused in v0.1.
    pub restitution: f32,
    /// How the body stores, conducts and sheds heat.
    ///
    /// Read only when the simulation's heat solve is on. The same coefficients
    /// a continuum uses, because a material does not conduct differently for
    /// being treated as rigid.
    pub thermal: crate::solid::Thermal,
    /// How stiff it is and how much load it carries before it gives way. Read
    /// by the stress solve; a body whose [`crate::Failure`] is `Never` only
    /// reports it.
    pub strength: Strength,
}

impl Default for Material {
    fn default() -> Self {
        Self::GENERIC
    }
}

impl Material {
    /// A neutral, roughly water-density material for scratch scenes.
    pub const GENERIC: Self = Self {
        density: 1000.0,
        friction: 0.5,
        restitution: 0.0,
        thermal: crate::solid::Thermal::GENERIC,
        strength: Strength::GENERIC,
    };

    // A first few entries of the cited library described in PLAN.md §12.
    // Densities are standard handbook values at room temperature; friction
    // coefficients are dry, self-paired, and approximate.
    /// Structural steel.
    pub const STEEL: Self = Self {
        density: 7850.0,
        friction: 0.6,
        restitution: 0.4,
        thermal: crate::solid::Thermal::STEEL,
        strength: Strength::STEEL,
    };
    /// Aluminium.
    pub const ALUMINIUM: Self = Self {
        density: 2700.0,
        friction: 0.5,
        restitution: 0.35,
        thermal: crate::solid::Thermal::ALUMINIUM,
        strength: Strength::ALUMINIUM,
    };
    /// Concrete.
    pub const CONCRETE: Self = Self {
        density: 2400.0,
        friction: 0.8,
        restitution: 0.1,
        thermal: crate::solid::Thermal::CONCRETE,
        strength: Strength::CONCRETE,
    };
    /// Softwood.
    pub const WOOD: Self = Self {
        density: 700.0,
        friction: 0.45,
        restitution: 0.3,
        thermal: crate::solid::Thermal::WOOD,
        strength: Strength::WOOD,
    };
    /// Stone: granite. Heavy, hard, and it chips.
    pub const STONE: Self = Self {
        density: 2700.0,
        friction: 0.7,
        restitution: 0.15,
        thermal: crate::solid::Thermal::STONE,
        strength: Strength::STONE,
    };
    /// Rubber: grippy and bouncy.
    pub const RUBBER: Self = Self {
        density: 1200.0,
        friction: 1.1,
        restitution: 0.8,
        thermal: crate::solid::Thermal::RUBBER,
        strength: Strength::RUBBER,
    };
    /// Ice: almost frictionless.
    pub const ICE: Self = Self {
        density: 917.0,
        friction: 0.05,
        restitution: 0.1,
        thermal: crate::solid::Thermal::ICE,
        strength: Strength::ICE,
    };
    /// Soda-lime glass, annealed. Stiff, hard, and brittle: it carries a great
    /// deal in compression and very little in tension, which is why it
    /// shatters.
    pub const GLASS: Self = Self {
        density: 2500.0,
        friction: 0.7,
        restitution: 0.5,
        thermal: crate::solid::Thermal::GLASS,
        strength: Strength::GLASS,
    };
    /// Water, for bodies that are meant to weigh what water weighs.
    pub const WATER: Self = Self {
        density: 1000.0,
        friction: 0.0,
        restitution: 0.0,
        thermal: crate::solid::Thermal::INERT,
        strength: Strength::NONE,
    };

    /// A material from its three properties.
    pub const fn new(density: f32, friction: f32, restitution: f32) -> Self {
        Self {
            density,
            friction,
            restitution,
            thermal: crate::solid::Thermal::GENERIC,
            strength: Strength::GENERIC,
        }
    }

    /// Builder: same material, conducting and storing heat differently.
    pub const fn with_thermal(mut self, thermal: crate::solid::Thermal) -> Self {
        self.thermal = thermal;
        self
    }

    /// Builder: same material, stiffer or stronger.
    pub const fn with_strength(mut self, strength: Strength) -> Self {
        self.strength = strength;
        self
    }

    /// Builder: same material at a different density.
    pub const fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }
}

/// Stiffness and strength of a rigid material.
///
/// **Real numbers**, unlike the continuum library's analogues. The continuum
/// is explicit, so its timestep is set by how fast waves cross the material,
/// and steel's 5 km/s is out of reach. The stress in a rigid body is solved as
/// a static problem instead, where stiffness only scales the answer: 200 GPa
/// steel costs exactly what 5 MPa rubber does.
///
/// Strengths are handbook figures at room temperature, and are lowered as the
/// material approaches its melting point — see [`Strength::at_temperature`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strength {
    /// Young's modulus, Pa. Zero for a material that carries no stress at
    /// all, which the stress solve skips.
    pub youngs_modulus: f32,
    /// Poisson's ratio, in `[0, 0.5)`.
    pub poissons_ratio: f32,
    /// Von Mises stress at which it stops springing back, Pa. Infinite for a
    /// brittle material, which cracks before it yields.
    pub yield_stress: f32,
    /// Tension it carries before it cracks, Pa: the ultimate tensile strength.
    pub tensile: f32,
    /// Compression it carries before it crushes, Pa.
    pub compressive: f32,
    /// Fibres: where the material is far stronger one way than the others,
    /// the strengths across that direction. `None` for a material that is
    /// the same every way. The strengths above are then the ones *along*
    /// the grain.
    ///
    /// Strength only. Stiffness is still one Young's modulus every way —
    /// wood is about twenty times stiffer along its grain than across it,
    /// which moves where its stress goes a little and how much less than
    /// what it can take. The stress is a body's, the grain only changes how
    /// much of it the material can carry and which way it parts.
    pub grain: Option<Grain>,
    /// How it comes apart when it breaks: the shape of the pieces. What
    /// separates a pane of glass from a block of stone of the same strength.
    pub pattern: Pattern,
}

/// How a material comes apart, beyond when: the pieces it leaves.
///
/// Two materials can part at the same stress and leave nothing alike. Glass
/// has nothing in it to stop a crack once one starts, so it runs, branches
/// and leaves shards, smallest where it was hit. Stone and concrete are
/// grains held in a matrix, and a crack wanders between the grains and
/// stops: what comes away is a chip off the corner that took the blow, and
/// only a heavy blow goes through. Wood's fibres split along their length and
/// splinter across it. A metal tears. A plain plastic cracks across, once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// One clean crack across the greatest tension; a second or third if
    /// most of it is overloaded. Plastics.
    #[default]
    Crack,
    /// Into many shards, clustered where it was struck: the crack branches
    /// as it runs. Glass.
    Shatter,
    /// A chip off where it was loaded, or a few blocky chunks if the whole
    /// of it was. Stone, concrete, ice.
    Chip,
    /// Along its grain into slats, or across it with ragged ends and
    /// splinters. Wood; it needs a [`Grain`].
    Splinter,
    /// Torn in two on the plane of greatest shear. Metals.
    Tear,
}

impl Pattern {
    /// Every pattern, with its label and one line on what it does.
    pub const ALL: [(Pattern, &'static str, &'static str); 5] = [
        (Pattern::Crack, "Cracks", "One clean crack across, the way a plastic goes."),
        (Pattern::Shatter, "Shatters", "Into shards, smallest where it was hit, the way glass goes."),
        (Pattern::Chip, "Chips", "A chip off the corner that took the blow, or blocky chunks under a heavy one, the way stone goes."),
        (Pattern::Splinter, "Splinters", "Along its grain into slats, or across it with splinters, the way wood goes."),
        (Pattern::Tear, "Tears", "In two, on the plane it was sheared on, the way a metal goes."),
    ];

    /// Its label: "Shatters".
    pub fn label(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(p, ..)| *p == self)
            .map(|(_, l, _)| *l)
            .unwrap_or("")
    }

    /// What it does, in a line.
    pub fn description(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(p, ..)| *p == self)
            .map(|(.., d)| *d)
            .unwrap_or("")
    }

    /// The pattern a material breaks in when nobody has said: wood's for one
    /// with a grain, a metal's for one that yields, a crack otherwise.
    pub fn implied_by(grain: bool, ductile: bool) -> Self {
        if grain {
            Pattern::Splinter
        } else if ductile {
            Pattern::Tear
        } else {
            Pattern::Crack
        }
    }
}

/// The strengths of a material with a grain, across it.
///
/// Wood is a bundle of hollow fibres glued together by a weaker matrix. Along
/// the fibres it is as strong as a light alloy for its weight; pulled across
/// them it parts at a fifteenth of that, and the fibres slide past each other
/// under a shear a sixth of it. Which of those three a load reaches first is
/// what decides how a piece of wood breaks: across, with splinters, when it
/// is bent; along, in long slats, when it is split or sheared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grain {
    /// Tension across the grain before the fibres pull apart, Pa.
    pub tensile_across: f32,
    /// Compression across the grain before it crushes, Pa. Wood crushed
    /// across its grain dents rather than breaks, so this counts towards
    /// how hard it is worked and never cracks it.
    pub compressive_across: f32,
    /// Shear along the grain before the fibres slide past each other, Pa.
    pub shear: f32,
}

impl Grain {
    /// Softwood — spruce, pine, fir — at about 12% moisture.
    pub const SOFTWOOD: Self = Self {
        tensile_across: 2.5e6,
        compressive_across: 5e6,
        shear: 7e6,
    };
}

/// How a stress state measures up against a material: see
/// [`Strength::assess`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assessment {
    /// Load over strength, every way the material can give: one is where it
    /// first yields, cracks or crushes.
    pub utilisation: f32,
    /// Load over the strength at which it parts — what breaking reads. Below
    /// [`Assessment::utilisation`] for a ductile metal, which bends a long way
    /// before it snaps, and for wood crushed across its grain, which dents.
    pub fracture: f32,
    /// The normal of the plane it would part on, in the frame the stress is
    /// in.
    pub normal: Vec3,
    /// How it would part.
    pub mode: Parting,
}

/// How a material parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Parting {
    /// Across the greatest tension: a clean crack, the way glass goes.
    #[default]
    Crack,
    /// On a plane of greatest shear: a metal torn, or a block crushed.
    Shear,
    /// Along the grain: split into slats, the way an axe splits a log.
    Split,
    /// Across the grain: snapped, with the fibres pulling out at different
    /// lengths, the way a bent plank goes.
    Snap,
}

impl Default for Strength {
    fn default() -> Self {
        Self::GENERIC
    }
}

impl Strength {
    /// A hard plastic, ABS or thereabouts.
    pub const GENERIC: Self = Self {
        youngs_modulus: 2.3e9,
        poissons_ratio: 0.35,
        yield_stress: 40e6,
        tensile: 45e6,
        compressive: 65e6,
        grain: None,
        pattern: Pattern::Crack,
    };
    /// Structural carbon steel (S275 or ASTM A36).
    pub const STEEL: Self = Self {
        youngs_modulus: 200e9,
        poissons_ratio: 0.3,
        yield_stress: 250e6,
        tensile: 400e6,
        compressive: 400e6,
        grain: None,
        pattern: Pattern::Tear,
    };
    /// Aluminium alloy 6061-T6.
    pub const ALUMINIUM: Self = Self {
        youngs_modulus: 69e9,
        poissons_ratio: 0.33,
        yield_stress: 276e6,
        tensile: 310e6,
        compressive: 310e6,
        grain: None,
        pattern: Pattern::Tear,
    };
    /// Plain concrete, C30. Ten times stronger crushed than pulled.
    pub const CONCRETE: Self = Self {
        youngs_modulus: 30e9,
        poissons_ratio: 0.2,
        yield_stress: f32::INFINITY,
        tensile: 3e6,
        compressive: 30e6,
        grain: None,
        pattern: Pattern::Chip,
    };
    /// Softwood. The strengths are along the grain, and
    /// [`Grain::SOFTWOOD`] gives the ones across it.
    pub const WOOD: Self = Self {
        youngs_modulus: 11e9,
        poissons_ratio: 0.3,
        yield_stress: f32::INFINITY,
        tensile: 40e6,
        compressive: 30e6,
        grain: Some(Grain::SOFTWOOD),
        pattern: Pattern::Splinter,
    };
    /// Granite, or any hard crystalline stone: strong in compression, weak in
    /// tension, and it chips.
    pub const STONE: Self = Self {
        youngs_modulus: 50e9,
        poissons_ratio: 0.25,
        yield_stress: f32::INFINITY,
        tensile: 10e6,
        compressive: 150e6,
        grain: None,
        pattern: Pattern::Chip,
    };
    /// Natural rubber. Stiff enough to carry a load and too stretchy to break
    /// here: it tears at around 15 MPa, but only after stretching to several
    /// times its length, which is far outside what a linear model can say. Read
    /// at face value, the model had a rubber block dropped ten metres
    /// shattering like glass; it never breaks instead.
    pub const RUBBER: Self = Self {
        youngs_modulus: 5e6,
        poissons_ratio: 0.45,
        yield_stress: f32::INFINITY,
        tensile: f32::INFINITY,
        compressive: f32::INFINITY,
        grain: None,
        pattern: Pattern::Crack,
    };
    /// Freshwater ice near its melting point.
    pub const ICE: Self = Self {
        youngs_modulus: 9e9,
        poissons_ratio: 0.33,
        yield_stress: f32::INFINITY,
        tensile: 1e6,
        compressive: 5e6,
        grain: None,
        pattern: Pattern::Chip,
    };
    /// Annealed soda-lime glass.
    pub const GLASS: Self = Self {
        youngs_modulus: 70e9,
        poissons_ratio: 0.22,
        yield_stress: f32::INFINITY,
        tensile: 40e6,
        compressive: 1e9,
        grain: None,
        pattern: Pattern::Shatter,
    };
    /// Carries no stress, and so never fails: water, or anything the stress
    /// solve should leave alone.
    pub const NONE: Self = Self {
        youngs_modulus: 0.0,
        poissons_ratio: 0.0,
        yield_stress: f32::INFINITY,
        tensile: f32::INFINITY,
        compressive: f32::INFINITY,
        grain: None,
        pattern: Pattern::Crack,
    };

    /// Whether the stress solve has anything to do for this material.
    #[inline]
    pub fn carries_load(&self) -> bool {
        self.youngs_modulus > 0.0 && self.youngs_modulus.is_finite()
    }

    /// Whether it yields before it cracks — a metal, rather than glass.
    #[inline]
    pub fn is_ductile(&self) -> bool {
        self.yield_stress < self.tensile
    }

    /// Speed of a stress wave along a bar, `sqrt(E / ρ)`, m/s.
    ///
    /// Sets how long an impact lasts: a bar struck on its end is in contact
    /// for the time a wave takes to cross it and come back, `2 L / c`.
    #[inline]
    pub fn bar_wave_speed(&self, density: f32) -> f32 {
        (self.youngs_modulus.max(0.0) / density.max(1e-6)).sqrt()
    }

    /// Fraction of the room-temperature strength left at `temperature`.
    ///
    /// Metals lose strength long before they melt — structural steel keeps
    /// about half at 600 °C and a tenth at 800 °C. Modelled on the homologous
    /// temperature `θ = T / T_melt`: full strength to `θ = 0.35`, then
    /// `((1 - θ) / 0.65)³` down to nothing at the melting point. Within a
    /// factor of two of the Eurocode reduction for steel across its range,
    /// which is a curve fitted to one material and not a measurement of any
    /// other.
    ///
    /// **Metals only** — the ductile materials, whose strength is dislocations
    /// moving, which heat makes easier. Ice at −20 °C is at 93% of its melting
    /// point and as strong as ice gets; on the metal curve it kept a
    /// six-hundredth of its strength and shattered on a 30 cm drop. A brittle
    /// material keeps its strength until it melts, and one that does not melt
    /// keeps it always.
    pub fn retained(&self, thermal: &crate::solid::Thermal, temperature: f32) -> f32 {
        if !thermal.melts() || thermal.melt_point <= 0.0 {
            return 1.0;
        }
        let theta = temperature / thermal.melt_point;
        if !self.is_ductile() {
            return if theta < 1.0 { 1.0 } else { 0.0 };
        }
        if theta <= 0.35 {
            1.0
        } else if theta >= 1.0 {
            0.0
        } else {
            ((1.0 - theta) / 0.65).powi(3)
        }
    }

    /// The same material's strengths at a temperature. Stiffness is left
    /// alone: it falls more slowly than strength, and the stress in a body
    /// held only by contacts does not depend on it.
    pub fn at_temperature(&self, thermal: &crate::solid::Thermal, temperature: f32) -> Self {
        let k = self.retained(thermal, temperature);
        Self {
            yield_stress: self.yield_stress * k,
            tensile: self.tensile * k,
            compressive: self.compressive * k,
            grain: self.grain.map(|g| Grain {
                tensile_across: g.tensile_across * k,
                compressive_across: g.compressive_across * k,
                shear: g.shear * k,
            }),
            ..*self
        }
    }

    /// How the stress `s` (`[xx, yy, zz, xy, yz, zx]`, Pa) measures up
    /// against this material, with its grain along the unit vector `grain`
    /// in the same frame. `grain` is ignored for a material without one.
    ///
    /// Without a grain this is [`Strength::utilisation`] and
    /// [`Strength::fracture_ratio`], parting across the largest principal
    /// stress. With one, the stress is read in the grain's frame:
    ///
    /// - the normal stress along the fibres against the strengths above,
    ///   tension snapping it across, and compression creasing it and — at
    ///   twice its compressive strength — kinking it apart on a plane at 45°;
    /// - the shear on the fibres against [`Grain::shear`], splitting it on
    ///   the plane through the grain that carries it;
    /// - the stress across the fibres, its two principal values against
    ///   [`Grain::tensile_across`] (split, across the larger) and
    ///   [`Grain::compressive_across`] (counted, but a dent, not a break).
    pub fn assess(&self, s: [f32; 6], grain: Vec3) -> Assessment {
        self.assess_parts(s, s, grain)
    }

    /// [`Strength::assess`], reading tension along the grain from `local`
    /// and everything else — crushing along it, shear on it, the stress
    /// across it — from `spread`. The two are the same stress except where
    /// the stress solve has a reason to judge them at different scales: a
    /// snap starts at the most stretched fibre, and the other ways wood gives
    /// need a body's worth of material to be loaded, not the few cells next
    /// to a contact. Without a grain only `local` is read.
    pub fn assess_parts(&self, local: [f32; 6], spread: [f32; 6], grain: Vec3) -> Assessment {
        let s = local;
        let Some(across) = self.grain.filter(|_| grain.length_squared() > 0.5) else {
            let (principal, directions) = crate::stress::principal(s);
            let vm = crate::stress::von_mises(s);
            let utilisation = self.utilisation(principal, vm);
            let fracture = self.fracture_ratio(principal, vm);
            let tension = ratio(principal[0], self.tensile);
            let crush = ratio(-principal[2], self.compressive);
            // Across the largest principal stress for a crack opening in
            // tension or a brittle split under compression; on the plane of
            // greatest shear for a ductile material torn by it.
            let torn = self.is_ductile()
                && self.tensile.is_finite()
                && vm / self.tensile > tension.max(crush);
            let (normal, mode) = if torn {
                (
                    (directions[0] + directions[2]).normalize_or_zero(),
                    Parting::Shear,
                )
            } else {
                (directions[0], Parting::Crack)
            };
            return Assessment {
                utilisation,
                fracture,
                normal: or_x(normal),
                mode,
            };
        };
        let g = grain.normalize();
        let stretched = apply(s, g).dot(g);
        let s = spread;
        let traction = apply(s, g);
        let along = traction.dot(g);
        let slide = traction - g * along;
        let tau = slide.length();
        // The stress across the fibres, in a basis of the plane they cross.
        let e1 = g.any_orthonormal();
        let e2 = g.cross(e1);
        let (a, b, c) = (
            apply(s, e1).dot(e1),
            apply(s, e2).dot(e2),
            apply(s, e1).dot(e2),
        );
        let mean = 0.5 * (a + b);
        let radius = (0.25 * (a - b) * (a - b) + c * c).sqrt();
        let (t1, t2) = (mean + radius, mean - radius);
        let angle = 0.5 * (2.0 * c).atan2(a - b);
        let d1 = e1 * angle.cos() + e2 * angle.sin();

        let pulled = ratio(stretched, self.tensile);
        // Fibres crushed along their length buckle into a crease and keep
        // carrying load: a plank bent past its compressive strength on top
        // breaks when the bottom lets go, in tension. Crushing alone parts
        // it — a post under a load it cannot hold — at about twice the
        // stress it starts to crease at.
        let creased = ratio(-along, self.compressive);
        let kinked = ratio(-along, KINK * self.compressive);
        let sheared = ratio(tau, across.shear);
        let opened = ratio(t1, across.tensile_across);
        let dented = ratio(-t2, across.compressive_across);
        let mut parts = [
            (pulled, g, Parting::Snap),
            (kinked, (g + d1).normalize_or_zero(), Parting::Shear),
            (
                sheared,
                if tau > 0.0 { slide / tau } else { d1 },
                Parting::Split,
            ),
            (opened, d1, Parting::Split),
        ];
        parts.sort_by(|x, y| y.0.total_cmp(&x.0));
        let (fracture, normal, mode) = parts[0];
        Assessment {
            utilisation: fracture.max(dented).max(creased),
            fracture,
            normal: or_x(normal),
            mode,
        }
    }

    /// Load over strength for a stress state: the largest of tension over the
    /// tensile strength, compression over the compressive strength, and — for
    /// a ductile material — von Mises over the yield stress. One is where the
    /// material first gives way.
    pub fn utilisation(&self, principal: [f32; 3], von_mises: f32) -> f32 {
        let mut u = ratio(principal[0], self.tensile).max(ratio(-principal[2], self.compressive));
        if self.is_ductile() {
            u = u.max(ratio(von_mises, self.yield_stress));
        }
        u
    }

    /// Load over the strength at which it actually parts: as
    /// [`Strength::utilisation`], but a ductile material is measured against
    /// its ultimate strength rather than its yield — steel bends a long way
    /// before it snaps.
    pub fn fracture_ratio(&self, principal: [f32; 3], von_mises: f32) -> f32 {
        let brittle = Self {
            yield_stress: f32::INFINITY,
            ..*self
        };
        let mut u = brittle.utilisation(principal, von_mises);
        if self.is_ductile() && self.tensile.is_finite() && self.tensile > 0.0 {
            u = u.max((von_mises / self.tensile).min(FAR_PAST));
        }
        u
    }
}

/// How many times its compressive strength wood crushed along its grain
/// takes before it parts.
const KINK: f32 = 2.0;

/// A load over a strength: zero for no load or no limit, and
/// [`FAR_PAST`] for a load on a material with no strength left — one heated
/// to its melting point. Finite, so the number can be drawn, graphed and
/// summed: an infinite load reached the trace and the graph's scale.
fn ratio(load: f32, strength: f32) -> f32 {
    if strength.is_finite() && strength > 0.0 {
        (load.max(0.0) / strength).min(FAR_PAST)
    } else if strength <= 0.0 && load > 0.0 {
        FAR_PAST
    } else {
        0.0
    }
}

/// The most a load over strength is ever reported as: far past breaking.
pub const FAR_PAST: f32 = 1e3;

/// `σ n` for a stress in `[xx, yy, zz, xy, yz, zx]`.
fn apply(s: [f32; 6], n: Vec3) -> Vec3 {
    let [xx, yy, zz, xy, yz, zx] = s;
    Vec3::new(
        xx * n.x + xy * n.y + zx * n.z,
        xy * n.x + yy * n.y + yz * n.z,
        zx * n.x + yz * n.y + zz * n.z,
    )
}

fn or_x(v: Vec3) -> Vec3 {
    if v.length_squared() > 0.5 {
        v
    } else {
        Vec3::X
    }
}

#[cfg(test)]
mod grain_tests {
    use super::*;

    const ALONG_X: Vec3 = Vec3::X;

    #[test]
    fn wood_is_strong_along_its_grain_and_weak_across_it() {
        let wood = Strength::WOOD;
        // 30 MPa of tension along the grain: three quarters of its strength.
        let a = wood.assess([30e6, 0.0, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert!((a.fracture - 0.75).abs() < 1e-4, "{a:?}");
        // The same pull across the grain splits it twelve times over, on a
        // plane through the grain.
        let b = wood.assess([0.0, 30e6, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert!((b.fracture - 12.0).abs() < 1e-3, "{b:?}");
        assert_eq!(b.mode, Parting::Split);
        assert!(b.normal.dot(ALONG_X).abs() < 1e-4 && b.normal.y.abs() > 0.999);
        // Pulled past its strength along the grain, it snaps across it.
        let c = wood.assess([50e6, 0.0, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert_eq!(c.mode, Parting::Snap);
        assert!(c.normal.x.abs() > 0.999);
    }

    #[test]
    fn shear_along_the_grain_splits_it_on_the_plane_that_carries_it() {
        // xy shear with the grain along x: the fibres slide over each other
        // on the plane whose normal is y.
        let a = Strength::WOOD.assess([0.0, 0.0, 0.0, 14e6, 0.0, 0.0], ALONG_X);
        assert_eq!(a.mode, Parting::Split);
        assert!((a.fracture - 2.0).abs() < 1e-3, "{a:?}");
        assert!(a.normal.y.abs() > 0.999, "{a:?}");
    }

    #[test]
    fn crushed_across_the_grain_it_dents_rather_than_breaks() {
        let a = Strength::WOOD.assess([0.0, -10e6, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert!((a.utilisation - 2.0).abs() < 1e-3, "{a:?}");
        assert!(a.fracture < 1.0, "{a:?}");
    }

    #[test]
    fn crushed_along_the_grain_it_creases_before_it_kinks() {
        let wood = Strength::WOOD;
        let creased = wood.assess([-45e6, 0.0, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert!(
            creased.utilisation > 1.0 && creased.fracture < 1.0,
            "{creased:?}"
        );
        let kinked = wood.assess([-70e6, 0.0, 0.0, 0.0, 0.0, 0.0], ALONG_X);
        assert!(
            kinked.fracture > 1.0 && kinked.mode == Parting::Shear,
            "{kinked:?}"
        );
    }

    #[test]
    fn a_material_without_a_grain_ignores_the_direction() {
        let s = [5e6, -2e6, 1e6, 3e6, 0.0, 1e6];
        let glass = Strength::GLASS;
        let a = glass.assess(s, Vec3::X);
        let b = glass.assess(s, Vec3::Y);
        assert_eq!(a, b);
        let (principal, _) = crate::stress::principal(s);
        let vm = crate::stress::von_mises(s);
        assert_eq!(a.fracture, glass.fracture_ratio(principal, vm));
    }
}
