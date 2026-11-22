//! Continuum solids: material points carrying deformation history.
//!
//! PLAN.md §11 states Advanced's world in one sentence — "not objects with
//! properties attached, but matter carrying state, distributed over a
//! lattice". This module is the *matter* half of that. The lattice lives in
//! `fusion-mpm`, is rebuilt every step, and is nowhere near here.
//!
//! ## Why this is a new kind of point and the fluid was not
//!
//! §5 warns against a universal simulation object, and the test it offers is
//! whether a system composes from the primitives or needs a new one. A fluid
//! passed that test: it is a [`ParticleSet`] plus a density constraint, and
//! every line already written against particles works on it unchanged.
//!
//! A material point does not, quite. It is still a [`ParticleSet`] — position,
//! velocity and mass mean exactly what they mean everywhere else in the engine
//! and are stored in the same arrays — but it carries something no other point
//! in Fusion carries: **history**. A fluid particle at a given position and
//! velocity is fully described; two material points at the same position and
//! velocity can be under quite different stress, because one of them has been
//! stretched and the other has not. That is the deformation gradient, and it
//! is why this module exists rather than a fourth constructor on `ParticleSet`.
//!
//! So the composition is kept where it holds and the addition is made
//! explicitly beside it: [`SolidSet`] *contains* a `ParticleSet` and adds four
//! parallel arrays of continuum state.
//!
//! ## What lives here and what does not
//!
//! Whatever is true about the solid when nothing is simulating it. The grid,
//! the transfers, the stress evaluation and the return mapping are the
//! method's, not the material's, and they live in `fusion-mpm`. What is here
//! is where the points are, how much they weigh, how far they have deformed,
//! and what they are made of.

use fusion_math::{Isometry, Mat3, Vec3};

use crate::fluid::fill;
use crate::particle::ParticleSet;
use crate::shape::Shape;

/// Reference to a body of continuum in a [`World`](crate::World).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SolidHandle(pub u32);

impl SolidHandle {
    /// The index this handle refers to.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The constitutive law a solid is solved under.
///
/// §11's claim is that "a steel bar and a puddle of water differ only in their
/// constitutive law", and this enum is where that claim is cashed. Everything
/// else about a material point — how it moves, how it collides, how it
/// transfers to and from the grid — is identical across every variant here.
///
/// Deliberately **not** `#[non_exhaustive]`, which is the opposite of what
/// `docs/API.md` rule 2 says for an enum a caller reads. The rule is right in
/// general and wrong here: a solver that meets a constitutive law it does not
/// implement and falls through to a `_` arm produces a material that looks
/// like a different material, and there is no diagnostic for that short of
/// somebody noticing that the sand behaves like jelly. Adding a variant should
/// stop the build in every place that has to learn the new law. Sand and the
/// mixtures §11 lists will be a breaking change, and that is the correct
/// price.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constitutive {
    /// Fixed corotated elasticity — Stomakhin et al.'s model, and the
    /// workhorse.
    ///
    /// `P = 2μ(F − R) + λ(J − 1) J F⁻ᵀ`, where `R` is the rotation from the
    /// polar decomposition. Corotated rather than linear because linear
    /// elasticity is only correct for small strains and produces visible
    /// ghost forces the moment something rotates: a bar swung about its end
    /// swells. Subtracting the rotation before measuring the strain is what
    /// removes that, and it is the reason this crate needed an SVD.
    ///
    /// ## It has no barrier against being crushed, and that is not a detail
    ///
    /// The volumetric pressure is `λ(J − 1)J`, which **peaks at `J = 0.5` and
    /// falls back to zero as `J → 0`**. Past that peak the material resists
    /// less the more it is compressed, which is a softening branch and
    /// therefore an instability: a point pushed beyond it keeps going, ends at
    /// `det F ≈ 0`, and has no restoring pressure left to come back with.
    ///
    /// Measured, because it is easy to say and easy to disbelieve: a 30 cm
    /// cube of a 10 kPa material dropped one metre onto a slab **flattens to
    /// 5 cm and stays there**. The identical material under
    /// [`Constitutive::NeoHookean`] — same stiffness, same scene, one term
    /// different — settles at 24.5 cm with a peak strain of 0.2. `λ ln J`
    /// diverges as the point flattens and this does not.
    ///
    /// So it is the right model for something stiff enough that `J` never
    /// approaches a half — porcelain, an alloy — and the wrong one for
    /// anything soft that might be hit hard. [`SolidMaterial::JELLY`] was
    /// corotated until that measurement.
    FixedCorotated,
    /// Compressible neo-Hookean, `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ`.
    ///
    /// Resists extreme compression harder than corotated does — the `ln J`
    /// term goes to negative infinity as the point flattens — which is what
    /// makes it the better choice for something being crushed and the worse
    /// one for something that may legitimately invert.
    NeoHookean,
    /// Elastoplastic snow: corotated elasticity with the elastic deformation
    /// clamped and the rest handed to a plastic multiplier that hardens the
    /// material as it packs.
    ///
    /// The one model here that has memory beyond `F`. Everything the material
    /// cannot return from accumulates in `SolidSet::plastic`, and the Lamé
    /// parameters are scaled by `exp(hardening * (1 - plastic))` — packed snow
    /// is stiffer than fresh snow, and this is that sentence as arithmetic.
    Snow {
        /// Compression past which deformation stops being recoverable, as a
        /// fraction. 0.025 means a point may be squashed to 97.5% of its
        /// length elastically and anything further is permanent.
        critical_compression: f32,
        /// Stretch past which the same applies, as a fraction.
        critical_stretch: f32,
        /// How sharply packing stiffens the material.
        hardening: f32,
    },
    /// Von Mises plasticity: corotated elasticity, and a shear strain past
    /// which the material flows instead of storing more. Metal.
    ///
    /// Measured in logarithmic strain, and only its deviatoric part: the
    /// shape changes, the volume does not, which is what a metal does when it
    /// yields and what [`Constitutive::Snow`] does not. Snow compacts — that
    /// is its whole point — and a steel ball given the snow law to yield
    /// with was crushed from a 92 cm sphere to a 2 cm sheet no wider than the
    /// ball, its volume simply gone. Under this law it squashes and spreads.
    VonMises {
        /// Shear strain the material holds elastically before it flows, as a
        /// fraction: its yield stress over its Young's modulus, near enough.
        yield_strain: f32,
    },
}

/// What a solid is made of.
///
/// §12 defines a material as a constitutive law plus its parameters, and this
/// is that pair. Separate from [`Material`](crate::Material) — which describes
/// a *rigid* body, where density, restitution and friction are the whole story
/// — because a continuum has no restitution and a rigid body has no Young's
/// modulus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidMaterial {
    /// Rest density, kg/m³. Sets the mass of each material point together with
    /// the sampling spacing.
    pub density: f32,
    /// Young's modulus, Pa. Stiffness in the everyday sense: how hard it is to
    /// stretch.
    ///
    /// **This is the number that sets the timestep**, and it is worth knowing
    /// that before choosing it. Elastic waves cross a material at `sqrt(E/ρ)`,
    /// and an explicit method has to resolve them: the CFL limit is roughly
    /// `dx / sqrt(E/ρ)`, so steel at 200 GPa needs a timestep near a
    /// microsecond and will not run at 240 Hz however many substeps are
    /// thrown at it. [`SolidMaterial::max_stable_dt`] answers the question
    /// directly rather than leaving it to be discovered as an explosion.
    pub youngs_modulus: f32,
    /// Poisson's ratio, dimensionless, in `[0, 0.5)`.
    ///
    /// How much a material bulges sideways when squashed. 0.5 is
    /// incompressible and is an asymptote rather than a value — λ has
    /// `1 - 2ν` in its denominator — so it is clamped below.
    pub poissons_ratio: f32,
    /// The law and whatever it needs.
    pub model: Constitutive,
    /// How it carries, conducts and responds to heat (v0.9).
    pub thermal: Thermal,
    /// When it stops holding together (v0.9).
    pub fracture: Fracture,
}

/// When a material tears, as a scalar damage law.
///
/// PLAN.md §11's third field: "Damage — a scalar per material point — fracture
/// that emerges instead of being authored." The emphasis is the point. There
/// is no crack geometry here, no pre-scored seam and no fracture pattern: a
/// point accumulates damage when it is pulled harder than it can bear, damaged
/// points carry less tension, so the load moves to their neighbours, which then
/// tear too. A crack is the path that process takes, and it is an *output*.
///
/// ## The criterion is stretch, not stress
///
/// Damage initiates on the largest principal stretch — the biggest singular
/// value of the deformation gradient. A stress criterion would be equally
/// defensible and is harder to write a number for: "this tears at 2% strain" is
/// a sentence somebody can check against a material they have held, and "this
/// tears at 40 kPa" is a sentence about a solver whose stiffnesses are already
/// scaled down to run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fracture {
    /// Stretch at which damage begins to accumulate. 1.02 is 2% strain.
    ///
    /// Infinite for a material that does not break, which is the default.
    pub onset: f32,
    /// Stretch at which the material has fully parted and carries no tension.
    pub failure: f32,
}

impl Default for Fracture {
    fn default() -> Self {
        Self::TOUGH
    }
}

impl Fracture {
    /// A material that never breaks, however hard it is pulled.
    ///
    /// The default. Damage is the one field here that changes a material's
    /// behaviour whether or not anybody asked for it, so it is off unless a
    /// material says otherwise — unlike heat, which a material can carry and
    /// ignore.
    pub const TOUGH: Self = Self {
        onset: f32::INFINITY,
        failure: f32::INFINITY,
    };

    /// True where this material can break at all.
    #[inline]
    pub fn breaks(&self) -> bool {
        self.onset.is_finite() && self.failure > self.onset
    }

    /// Damage implied by a maximum principal stretch, in `[0, 1]`.
    ///
    /// Linear softening between onset and failure. Linear because the shape of
    /// the softening curve is a material property nobody has measured for the
    /// analogues this library ships, and a straight line between two numbers
    /// that *are* stated is more honest than a curve fitted to nothing.
    #[inline]
    pub fn damage_at(&self, stretch: f32) -> f32 {
        if !self.breaks() || stretch <= self.onset {
            return 0.0;
        }
        ((stretch - self.onset) / (self.failure - self.onset)).clamp(0.0, 1.0)
    }
}

/// Room temperature, Kelvin. What a solid starts at unless told otherwise.
pub const AMBIENT: f32 = 293.15;

/// A material's thermal properties.
///
/// PLAN.md §11's table puts temperature and phase on the same lattice as the
/// momentum solve, and the reason it can is that adding physics to a continuum
/// is adding *fields* rather than adding systems. These are the coefficients
/// that field needs.
///
/// ## Why heat is stored as enthalpy and not as temperature
///
/// A melting material absorbs energy without getting hotter, so temperature is
/// not a state variable across a phase change — it is flat through the whole
/// of it, and a solver that integrates temperature has nowhere to put the
/// latent heat. Enthalpy is monotonic in energy and stays a state variable, so
/// the phase falls out of it rather than needing to be tracked separately.
/// §11 names the enthalpy formulation for exactly this reason.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thermal {
    /// Specific heat capacity, J/(kg K). How much energy a kelvin costs.
    pub specific_heat: f32,
    /// Thermal conductivity, W/(m K). How fast heat crosses the material.
    pub conductivity: f32,
    /// Linear expansion coefficient, 1/K. How much it grows when heated.
    pub expansion: f32,
    /// Melting point, Kelvin. Infinite for something that does not melt.
    pub melt_point: f32,
    /// Latent heat of fusion, J/kg — the energy a phase change absorbs at
    /// constant temperature.
    pub latent_heat: f32,
    /// Fraction of stiffness lost at the melting point, in `[0, 1]`.
    ///
    /// The coupling that makes §11's demo work: a bar struck hard enough heats
    /// at the impact site, and what happens next is that it *softens there*, so
    /// the next blow deforms the same spot further. Neither half is authored —
    /// the heat comes from plastic work and the softening comes from this
    /// number.
    pub softening: f32,
    /// Heat generated per joule of plastic work, in `[0, 1]`.
    ///
    /// The Taylor–Quinney coefficient, and it is close to one for a metal:
    /// almost all the work spent on permanent deformation ends up as heat
    /// rather than as stored dislocation energy. This is the term that makes
    /// the temperature rise *emerge* from the mechanics instead of being
    /// painted on.
    pub plastic_heating: f32,
    /// Surface emissivity, in `[0, 1]`: how much of a black body's thermal
    /// radiation the surface gives off.
    ///
    /// Radiation is `epsilon sigma (T^4 - T_air^4)` per square metre, so it is
    /// negligible near room temperature and dominant once something glows —
    /// a steel block at 1000 K sheds about 45 kW/m^2 this way against a few
    /// hundred by convection. Read by the rigid-body heat solve; the continuum
    /// does not radiate yet.
    pub emissivity: f32,
}

impl Default for Thermal {
    fn default() -> Self {
        Self::INERT
    }
}

impl Thermal {
    /// A material that carries heat and is not changed by it.
    ///
    /// The default, so that adding a solid to a scene does not silently opt
    /// into a thermal solve nobody asked for. Water's specific heat and a
    /// conductivity in the range of most soft matter, but no expansion, no
    /// melting point and no softening: heat diffuses through it and does
    /// nothing.
    pub const INERT: Self = Self {
        specific_heat: 4180.0,
        conductivity: 0.6,
        expansion: 0.0,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.0,
        emissivity: 0.95,
    };

    /// Ice, which melts at a temperature a scene can actually reach.
    pub const ICE: Self = Self {
        specific_heat: 2100.0,
        conductivity: 2.2,
        expansion: 5.1e-5,
        melt_point: 273.15,
        latent_heat: 334_000.0,
        softening: 0.98,
        plastic_heating: 0.6,
        emissivity: 0.97,
    };

    /// A ductile metal analogue: conducts well, expands, softens as it heats,
    /// and turns most of its plastic work into heat.
    ///
    /// The melting point is a real one for a soft alloy rather than steel's,
    /// for the same reason [`SolidMaterial::PUTTY`] is not called steel: the
    /// mechanical timestep is set by the stiffness, and this material has to
    /// be one the solver can carry.
    pub const ALLOY: Self = Self {
        specific_heat: 900.0,
        conductivity: 50.0,
        expansion: 2.3e-5,
        melt_point: 456.0,
        latent_heat: 60_000.0,
        softening: 0.95,
        plastic_heating: 0.9,
        emissivity: 0.3,
    };

    // ---- rigid-body presets -------------------------------------------------
    //
    // Handbook values at room temperature, for the rigid materials in
    // `Material`. A rigid body never deforms, so `softening` is zero and
    // `plastic_heating` is the metal or non-metal figure for when the same
    // material is used as a continuum. Something that chars or decomposes
    // before it melts has an infinite melting point here: it does not melt.

    /// A neutral solid: stone-like heat capacity and conductivity.
    pub const GENERIC: Self = Self {
        specific_heat: 1000.0,
        conductivity: 1.0,
        expansion: 1.0e-5,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.9,
    };

    /// Structural carbon steel, lightly oxidized.
    pub const STEEL: Self = Self {
        specific_heat: 490.0,
        conductivity: 50.0,
        expansion: 1.2e-5,
        melt_point: 1783.0,
        latent_heat: 270_000.0,
        softening: 0.0,
        plastic_heating: 0.9,
        emissivity: 0.8,
    };

    /// Aluminium, bare.
    pub const ALUMINIUM: Self = Self {
        specific_heat: 900.0,
        conductivity: 205.0,
        expansion: 2.3e-5,
        melt_point: 933.0,
        latent_heat: 397_000.0,
        softening: 0.0,
        plastic_heating: 0.9,
        emissivity: 0.2,
    };

    /// Softwood, across the grain — pine or spruce, to go with
    /// `Material::WOOD`'s density. Chars rather than melting.
    pub const WOOD: Self = Self {
        specific_heat: 1700.0,
        conductivity: 0.12,
        expansion: 5.0e-6,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.9,
    };

    /// Concrete.
    pub const CONCRETE: Self = Self {
        specific_heat: 880.0,
        conductivity: 1.7,
        expansion: 1.2e-5,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.9,
    };

    /// Granite. It melts, at about 1 500 K, but long after anything here
    /// would have broken it, so it is left as not melting.
    pub const STONE: Self = Self {
        specific_heat: 790.0,
        conductivity: 2.8,
        expansion: 8.0e-6,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.9,
    };

    /// Natural rubber. Decomposes rather than melting.
    pub const RUBBER: Self = Self {
        specific_heat: 2000.0,
        conductivity: 0.16,
        expansion: 7.7e-5,
        melt_point: f32::INFINITY,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.9,
    };

    /// Soda-lime glass. It has no melting point in the sense a crystal does —
    /// it softens over a range — so the figure is where it flows, and the
    /// latent heat is zero.
    pub const GLASS: Self = Self {
        specific_heat: 840.0,
        conductivity: 1.0,
        expansion: 9.0e-6,
        melt_point: 1400.0,
        latent_heat: 0.0,
        softening: 0.0,
        plastic_heating: 0.6,
        emissivity: 0.92,
    };

    /// What something of this material is at when nobody has heated or
    /// cooled it: room temperature, or twenty kelvin below its own melting
    /// point if that is lower. See [`SolidMaterial::natural_temperature`].
    pub fn resting_temperature(&self) -> f32 {
        if self.melts() {
            AMBIENT.min(self.melt_point - 20.0)
        } else {
            AMBIENT
        }
    }

    /// True where this material can melt at all.
    #[inline]
    pub fn melts(&self) -> bool {
        self.melt_point.is_finite() && self.latent_heat >= 0.0
    }

    /// Thermal diffusivity, m²/s — `k / (ρ c)`, the coefficient in the heat
    /// equation.
    #[inline]
    pub fn diffusivity(&self, density: f32) -> f32 {
        let denominator = density.max(1e-6) * self.specific_heat.max(1e-6);
        self.conductivity / denominator
    }

    /// The specific enthalpy a temperature corresponds to, J/kg, for a
    /// material that has not begun to melt.
    ///
    /// Returned in double for the reason [`SolidSet::enthalpy`] is stored in
    /// double: this number is around a quarter of a million and the increments
    /// added to it are thousandths.
    #[inline]
    pub fn enthalpy_at(&self, temperature: f32) -> f64 {
        let base = self.specific_heat as f64 * temperature as f64;
        if !self.melts() || temperature < self.melt_point {
            base
        } else {
            base + self.latent_heat as f64
        }
    }

    /// Temperature and melted fraction implied by a specific enthalpy.
    ///
    /// The whole of the enthalpy formulation, in seven lines: below the
    /// solidus the material is solid and its temperature is `h / c`; between
    /// the solidus and the liquidus it sits *at* the melting point while the
    /// latent heat is absorbed; above it, it is liquid and heating again.
    /// Both are returned in double, and the temperature is the one that
    /// matters: the diffusion pass works on *differences* between neighbouring
    /// temperatures, and near 300 K the gap between representable `f32` values
    /// is 3 x 10^-5 K. A gradient finer than that is not small in single
    /// precision, it is absent.
    #[inline]
    pub fn state_of(&self, enthalpy: f64) -> (f64, f64) {
        let c = (self.specific_heat as f64).max(1e-6);
        if !self.melts() {
            return (enthalpy / c, 0.0);
        }
        let solidus = c * self.melt_point as f64;
        let latent = self.latent_heat as f64;
        if enthalpy <= solidus {
            (enthalpy / c, 0.0)
        } else if enthalpy < solidus + latent {
            let melted = if latent > 0.0 {
                (enthalpy - solidus) / latent
            } else {
                1.0
            };
            (self.melt_point as f64, melted.clamp(0.0, 1.0))
        } else {
            ((enthalpy - latent) / c, 1.0)
        }
    }
}

impl Default for SolidMaterial {
    fn default() -> Self {
        Self::JELLY
    }
}

impl SolidMaterial {
    /// A soft elastic solid that holds its shape and wobbles.
    ///
    /// The default, and deliberately the softest useful thing in the library:
    /// a first solid that explodes teaches nothing about MPM and a great deal
    /// about the CFL condition.
    ///
    /// **Neo-Hookean rather than corotated**, which for the softest material
    /// in the library is a correctness choice and not a taste one. See
    /// [`Constitutive::FixedCorotated`]: its volumetric pressure peaks at half
    /// volume and falls away after, so the material most likely to be crushed
    /// past that peak is the softest one. Dropped a metre onto a slab, this
    /// block flattened to 5 cm and stayed flat; with the model below and
    /// nothing else changed it settles at 24.5 cm and wobbles.
    pub const JELLY: Self = Self {
        density: 1000.0,
        youngs_modulus: 1.0e4,
        poissons_ratio: 0.3,
        model: Constitutive::NeoHookean,
        thermal: Thermal::INERT,
        fracture: Fracture::TOUGH,
    };

    /// Stiffer and bouncier — a solid that keeps its edges under its own
    /// weight.
    pub const RUBBER: Self = Self {
        density: 1100.0,
        youngs_modulus: 5.0e5,
        poissons_ratio: 0.45,
        model: Constitutive::NeoHookean,
        thermal: Thermal::INERT,
        fracture: Fracture::TOUGH,
    };

    /// Very compliant and very compressible, for something that squashes flat
    /// and comes back.
    pub const FOAM: Self = Self {
        density: 200.0,
        youngs_modulus: 3.0e3,
        poissons_ratio: 0.1,
        model: Constitutive::NeoHookean,
        thermal: Thermal::INERT,
        fracture: Fracture::TOUGH,
    };

    /// Elastoplastic snow, from Stomakhin et al.'s parameters.
    pub const SNOW: Self = Self {
        density: 400.0,
        youngs_modulus: 1.4e5,
        poissons_ratio: 0.2,
        model: Constitutive::Snow {
            critical_compression: 0.025,
            critical_stretch: 0.0075,
            hardening: 10.0,
        },
        // Snow is ice, and ice melts at a temperature a scene reaches.
        thermal: Thermal::ICE,
        fracture: Fracture::TOUGH,
    };

    /// A ductile solid that keeps the shape it is dented into.
    ///
    /// Not steel, and named so that nobody expects it to be: steel's real
    /// modulus is 200 GPa, which is four orders of magnitude past what an
    /// explicit method resolves at a graphics timestep. This is a soft metal
    /// analogue — the *behaviour* of yielding and holding a dent, at a
    /// stiffness the solver can carry. §9's rule applied to a material rather
    /// than to a solver: say which tier it is in.
    pub const PUTTY: Self = Self {
        density: 1800.0,
        youngs_modulus: 8.0e5,
        poissons_ratio: 0.3,
        // Clay flows and keeps its volume: von Mises. It was the snow law,
        // which compacts, and a putty ball dropped 3 m ended a third of its
        // height and no wider — two thirds of it gone. (Earlier still it
        // borrowed snow's tensile threshold and had no tensile strength at
        // all; a yield surface that holds in shear is what it needed.)
        // About 50 kPa, plasticine's order: enough to hold a ball up under
        // its own weight, which 10 kPa was not — a 1 m ball slumped into a
        // pancake four metres across and tore into blobs once it was
        // thinner than a cell.
        model: Constitutive::VonMises { yield_strain: 0.06 },
        thermal: Thermal::ALLOY,
        fracture: Fracture::TOUGH,
    };

    /// A ductile metal analogue that heats, softens and melts.
    ///
    /// §11's demo material: "a metal bar is struck, heats at the impact site,
    /// softens, yields, necks and tears". Every one of those verbs is a
    /// consequence of a number in this constant rather than an effect — the
    /// heating is `Thermal::plastic_heating` on the work the yield does, the
    /// softening is `Thermal::softening` on the temperature that produces, and
    /// the necking is what a softer spot does when the bar is pulled.
    pub const ALLOY: Self = Self {
        density: 2700.0,
        youngs_modulus: 2.0e6,
        poissons_ratio: 0.33,
        // Metal yields in shear and keeps its volume: von Mises. Under the
        // snow law it compacted instead, squashed to a third of its height
        // without spreading. Yields at 100 kPa: an analogue's stress at an
        // analogue's stiffness, firm enough to stand and to be dented
        // rather than splashed.
        model: Constitutive::VonMises { yield_strain: 0.05 },
        thermal: Thermal::ALLOY,
        fracture: Fracture::TOUGH,
    };

    /// A brittle solid that shatters rather than deforming.
    ///
    /// The material damage was built for: stiff enough to hold its shape, and
    /// with a failure strain under a percent, so what it does under load is
    /// crack rather than squash. Its fracture pattern is not authored anywhere
    /// — it is wherever the tension went.
    pub const PORCELAIN: Self = Self {
        density: 2300.0,
        youngs_modulus: 5.0e6,
        poissons_ratio: 0.22,
        model: Constitutive::FixedCorotated,
        thermal: Thermal::INERT,
        fracture: Fracture {
            // A percent of strain to start and three to part.
            //
            // The first values were four times tighter, and a block landing at
            // one metre per second shattered completely. That is not the
            // criterion being wrong so much as being applied to the wrong
            // number: the largest principal stretch of a *squashed* block is
            // its lateral bulge, and Poisson's ratio turns a 5% squash into a
            // 1% stretch sideways. A brittle solid really does fail that way —
            // it splits along the load rather than crushing — but a failure
            // strain has to be set knowing that is what will reach it first.
            onset: 1.01,
            failure: 1.03,
        },
    };

    /// Every preset, with the name to show for it.
    pub const LIBRARY: [(&'static str, Self); 7] = [
        ("Jelly", Self::JELLY),
        ("Rubber", Self::RUBBER),
        ("Foam", Self::FOAM),
        ("Snow", Self::SNOW),
        ("Putty", Self::PUTTY),
        ("Alloy", Self::ALLOY),
        ("Porcelain", Self::PORCELAIN),
    ];

    /// The Lamé parameters `(μ, λ)` this material implies.
    #[inline]
    pub fn lame(&self) -> (f32, f32) {
        let e = self.youngs_modulus.max(0.0);
        let nu = self.poissons_ratio.clamp(0.0, 0.49);
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (mu, lambda)
    }

    /// The Lamé parameters at a temperature and melted fraction.
    ///
    /// Thermal softening, which is one line of arithmetic and is the coupling
    /// that turns a heat field from a colour into physics. A material at its
    /// melting point keeps `1 - softening` of its stiffness, and one that has
    /// melted through keeps a hundredth of that — not zero, because a material
    /// point with no stiffness at all has no restoring force, and MPM's grid
    /// will happily let it interpenetrate itself.
    ///
    /// Below the reference temperature nothing happens: a cold material is not
    /// stiffer than a room-temperature one here, which is a real simplification
    /// and the honest place to make it.
    pub fn lame_at(&self, temperature: f32, melted: f32) -> (f32, f32) {
        let (mu, lambda) = self.lame();
        if self.thermal.softening <= 0.0 || !self.thermal.melt_point.is_finite() {
            return (mu, lambda);
        }
        let span = (self.thermal.melt_point - AMBIENT).max(1.0);
        let heated = ((temperature - AMBIENT) / span).clamp(0.0, 1.0);
        let solid_scale = 1.0 - self.thermal.softening * heated;
        let scale = solid_scale * (1.0 - melted.clamp(0.0, 1.0) * 0.99);
        (mu * scale.max(1e-4), lambda * scale.max(1e-4))
    }

    /// The temperature this material is placed at when nothing says otherwise.
    ///
    /// Room temperature, except for something that would not survive it. Snow
    /// at 20 °C melts, and it is right that it does — but a preset that turns
    /// to slush the instant it is placed is a bad preset, not a physics
    /// finding. Anything with a melting point below ambient is placed twenty
    /// kelvin below *its own* melting point instead.
    ///
    /// Found by giving `SNOW` real ice properties and watching it collapse:
    /// it had been "snow that cannot melt" through v0.8, and the first honest
    /// thermal model of it melted it where it stood.
    pub fn natural_temperature(&self) -> f32 {
        self.thermal.resting_temperature()
    }

    /// Speed of an elastic wave in this material, m/s.
    #[inline]
    pub fn wave_speed(&self) -> f32 {
        let (mu, lambda) = self.lame();
        let density = self.density.max(1e-6);
        ((lambda + 2.0 * mu) / density).sqrt()
    }

    /// The largest timestep an explicit method can take on a grid of this
    /// cell size, seconds. The CFL condition, stated rather than discovered.
    ///
    /// A step longer than this does not degrade gracefully — information has
    /// to cross a cell in one step or the scheme is unconditionally unstable,
    /// and what that looks like is a solid that shivers and then leaves.
    #[inline]
    pub fn max_stable_dt(&self, cell_size: f32) -> f32 {
        let speed = self.wave_speed();
        if speed <= 0.0 {
            f32::INFINITY
        } else {
            // The 0.4 is the usual explicit-MPM safety factor: the bound is
            // derived for a linear wave and the constitutive models here are
            // not linear.
            0.4 * cell_size / speed
        }
    }
}

/// How a solid is sampled, and what it is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidParams {
    /// Lattice spacing the material points were sampled at, metres. Point mass
    /// follows as `density * spacing³`.
    pub spacing: f32,
    /// Background grid cell size, metres.
    ///
    /// Twice the spacing by default, which puts eight material points in a
    /// cell. Fewer than about four and the grid velocity is a poor average, so
    /// the solid develops a cell-shaped shimmer; many more and the material is
    /// paying for resolution the grid will not carry.
    pub cell_size: f32,
    /// The constitutive law and its parameters.
    pub material: SolidMaterial,
    /// Fibres running through it: a structure that makes it stiffer pulled
    /// one way than another, the way wood, muscle or a woven sheet are. Read
    /// by the unified continuum solver; the classic one treats the material
    /// as the same every way.
    pub fibre: Option<Fibre>,
}

/// Fibres in a continuum: a direction, fixed in the material as it was
/// sampled, along which it resists being stretched harder than its
/// constitutive law alone says.
///
/// The fibre term is the standard fibre-reinforced one: with `a` the rest
/// direction and `f = F a` where the fibres are now, it stores
/// `k/4 (|f|² - 1)²` of energy for `|f| > 1`, so its Kirchhoff stress is
/// `k (|f|² - 1) f fᵀ`. Tension only: a fibre pushed along its length buckles
/// and carries nothing, which is why a rope cannot be pushed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fibre {
    /// Unit direction of the fibres in the material's rest frame.
    pub direction: Vec3,
    /// How much stiffer it is along the fibres, as a multiple of Young's
    /// modulus: `k = stiffness · E`.
    pub stiffness: f32,
}

impl Default for SolidParams {
    fn default() -> Self {
        Self::at(0.02)
    }
}

impl SolidParams {
    /// Default parameters at a given sampling spacing.
    pub fn at(spacing: f32) -> Self {
        Self {
            spacing,
            cell_size: 2.0 * spacing,
            material: SolidMaterial::default(),
            fibre: None,
        }
    }

    /// Builder: use this material.
    pub fn of(mut self, material: SolidMaterial) -> Self {
        self.material = material;
        self
    }

    /// Mass of one material point, kg.
    #[inline]
    pub fn point_mass(&self) -> f32 {
        self.material.density * self.point_volume()
    }

    /// Volume one material point stands for, m³.
    #[inline]
    pub fn point_volume(&self) -> f32 {
        let s = self.spacing.max(1e-6);
        s * s * s
    }
}

/// One body of continuum: its material points and their history.
#[derive(Clone, Debug, Default)]
pub struct SolidSet {
    /// The point masses. The same type cloth and fluid use (PLAN.md §5).
    pub particles: ParticleSet,
    /// Deformation gradient per point, `F`.
    ///
    /// The map from the material's rest configuration to where it is now.
    /// Identity is undeformed; `det F` is the volume ratio, so 0.9 is a point
    /// squashed to nine tenths of its rest volume. This is the state that
    /// makes a solid a solid rather than a cloud.
    pub deformation: Vec<Mat3>,
    /// Affine velocity per point, `C` — the local velocity gradient.
    ///
    /// APIC's contribution, and the reason this is not the 1990s method. PIC
    /// transfers only the velocity and loses the local rotation every step, so
    /// a spinning block grinds to a halt; FLIP transfers the difference and
    /// keeps noise no filter removes. Carrying the first moment as well is
    /// what makes the transfer both stable and non-dissipative.
    pub affine: Vec<Mat3>,
    /// Rest volume per point, m³. Fixed at sampling; the *current* volume is
    /// this times `det F`.
    pub volume: Vec<f32>,
    /// Plastic volume ratio per point, for the models that have one.
    ///
    /// One for a point that has never yielded. Below one for a point that has
    /// been permanently compacted, which is what hardens it.
    pub plastic: Vec<f32>,
    /// Accumulated damage per point, in `[0, 1]` (v0.9).
    ///
    /// Zero is intact and one has parted. Monotone: a material does not heal,
    /// which is what makes a crack a thing that happened rather than a thing
    /// that is currently true.
    pub damage: Vec<f32>,
    /// Specific enthalpy per point, J/kg (v0.9).
    ///
    /// Heat is stored here rather than as a temperature, because temperature
    /// stops being a state variable the moment anything melts: it is flat
    /// through the whole phase change while energy pours in. See [`Thermal`].
    /// [`SolidSet::temperature`] and [`SolidSet::melted`] derive both from it.
    ///
    /// **In double, and that is not caution.** PLAN.md §9 asks for precision
    /// to be a per-solver choice and names "an FP64 thermal solve coupled to
    /// FP32 material points" as the common case; this field is why. Enthalpy
    /// is an absolute quantity of order 10^5 J/kg and the increments the
    /// solver adds to it are of order 10^-3, so single precision cannot
    /// represent the sum. Measured: two watts into half a kilogram, which is
    /// +0.044 K after ten seconds, rounds to *nothing at all* in `f32` — every
    /// increment lands below half an ULP and the material never warms. At
    /// twenty watts it warms by 0.5 K where the answer is 0.444 K, a 12%
    /// error with the sign always the same, because repeated rounding of a
    /// constant increment is a bias rather than noise.
    pub enthalpy: Vec<f64>,
    /// The parameters this solid was built with.
    pub params: SolidParams,
    /// The temperature every point was created at, Kelvin. Kept so that
    /// [`SolidSet::push`] on an existing solid agrees with the rest of it.
    pub temperature_at_rest: f32,
}

impl SolidSet {
    /// An empty solid with these parameters.
    pub fn new(params: SolidParams) -> Self {
        Self {
            particles: ParticleSet::new(),
            deformation: Vec::new(),
            affine: Vec::new(),
            volume: Vec::new(),
            plastic: Vec::new(),
            damage: Vec::new(),
            enthalpy: Vec::new(),
            params,
            temperature_at_rest: AMBIENT,
        }
    }

    /// How many material points it holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// True where it holds none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Add one undeformed material point at the solid's own mass.
    pub fn push(&mut self, position: Vec3) {
        self.particles.push(
            position,
            self.params.point_mass(),
            self.params.spacing * 0.5,
        );
        self.deformation.push(Mat3::IDENTITY);
        self.affine.push(Mat3::ZERO);
        self.volume.push(self.params.point_volume());
        self.plastic.push(1.0);
        self.damage.push(0.0);
        self.enthalpy.push(
            self.params
                .material
                .thermal
                .enthalpy_at(self.temperature_at_rest),
        );
    }

    /// Temperature of one material point, Kelvin.
    ///
    /// Narrowed to `f32` on the way out: this is what a colour map and a
    /// readout want, and neither is differencing it. The solver uses
    /// [`Thermal::state_of`] directly and keeps the width.
    #[inline]
    pub fn temperature(&self, i: usize) -> f32 {
        self.params.material.thermal.state_of(self.enthalpy[i]).0 as f32
    }

    /// Melted fraction of one material point, in `[0, 1]`.
    #[inline]
    pub fn melted(&self, i: usize) -> f32 {
        self.params.material.thermal.state_of(self.enthalpy[i]).1 as f32
    }

    /// Hottest point in the solid, Kelvin.
    pub fn peak_temperature(&self) -> f32 {
        (0..self.len())
            .map(|i| self.temperature(i))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Fraction of the solid that has melted, by mass.
    pub fn melted_fraction(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let sum: f32 = (0..self.len()).map(|i| self.melted(i)).sum();
        sum / self.len() as f32
    }

    /// Worst damage anywhere in the solid, in `[0, 1]`.
    pub fn peak_damage(&self) -> f32 {
        self.damage.iter().copied().fold(0.0, f32::max)
    }

    /// Fraction of material points that have fully parted.
    ///
    /// The number that says a solid has broken rather than merely been
    /// stressed, and the one to watch converge — or not — with resolution.
    pub fn broken_fraction(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let broken = self.damage.iter().filter(|d| **d >= 0.999).count();
        broken as f32 / self.len() as f32
    }

    /// Total heat the solid holds, J.
    ///
    /// The conserved quantity of the thermal solve: diffusion moves it between
    /// points and may not create or destroy it, so a run where this drifts
    /// without a source has a bug in it.
    pub fn heat(&self) -> f64 {
        let mass = self.params.point_mass() as f64;
        self.enthalpy.iter().map(|h| h * mass).sum()
    }

    /// Total mass, kg. Constant: MPM creates and destroys no matter, so this
    /// is worth asserting on the way the fluid's is.
    pub fn mass(&self) -> f32 {
        self.particles
            .inv_mass
            .iter()
            .filter(|&&im| im > 0.0)
            .map(|&im| 1.0 / im as f64)
            .sum::<f64>() as f32
    }

    /// Total linear momentum, kg m/s.
    pub fn momentum(&self) -> Vec3 {
        self.particles
            .velocity
            .iter()
            .zip(&self.particles.inv_mass)
            .filter(|(_, im)| **im > 0.0)
            .fold(Vec3::ZERO, |sum, (v, im)| sum + *v / *im)
    }

    /// Current volume, m³ — rest volume carried through the deformation.
    ///
    /// Compared against `mass / density` this is the conservation check with
    /// teeth: an elastic solid at rest under no load has to come back to the
    /// volume it was sampled at.
    pub fn current_volume(&self) -> f32 {
        self.deformation
            .iter()
            .zip(&self.volume)
            .map(|(f, v)| v * f.determinant())
            .sum()
    }

    /// The smallest volume ratio anywhere in the solid, `min(det F)`.
    ///
    /// Separate from [`SolidSet::peak_strain`], which takes an absolute value
    /// and so cannot tell a point stretched to twice its volume from one
    /// crushed to nothing. Only one of those is a failure the solve may not
    /// come back from — see [`Constitutive::FixedCorotated`] on why a crushed
    /// point can run out of restoring pressure.
    pub fn least_volume(&self) -> f32 {
        self.deformation
            .iter()
            .map(|f| f.determinant())
            .fold(f32::INFINITY, f32::min)
    }

    /// Largest `|det F - 1|` in the solid, as a fraction.
    ///
    /// The diagnostic that says whether the solve is holding: a well-resolved
    /// elastic solid under gravity sits at a few percent, and a number
    /// climbing past one is a solid on its way to leaving.
    pub fn peak_strain(&self) -> f32 {
        self.deformation
            .iter()
            .map(|f| (f.determinant() - 1.0).abs())
            .fold(0.0, f32::max)
    }
}

/// A solid to be created, as data.
///
/// PLAN.md §16: the scene is small and the state is not. This is the small
/// half — a shape, a pose and a material, from which every material point in
/// the body is derived.
#[derive(Clone, Debug, PartialEq)]
pub struct SolidDesc {
    /// Region to fill, sampled through its SDF like a fluid's.
    pub shape: Shape,
    /// Where it is and how it is oriented.
    pub pose: Isometry,
    /// Spacing, cell size and material.
    pub params: SolidParams,
    /// Velocity every point starts with.
    pub velocity: Vec3,
    /// Temperature every point starts at, Kelvin.
    ///
    /// `None` means [`SolidMaterial::natural_temperature`] — an `Option` and
    /// not a defaulted `f32`, so that "I did not say" and "I said room
    /// temperature" stay different answers and neither depends on the order
    /// the builders were called in.
    pub temperature: Option<f32>,
}

impl SolidDesc {
    /// A solid filling the given shape.
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            pose: Isometry::IDENTITY,
            params: SolidParams::default(),
            velocity: Vec3::ZERO,
            temperature: None,
        }
    }

    /// A rectangular solid, from its full size.
    pub fn block(size: Vec3) -> Self {
        Self::new(Shape::cuboid(size))
    }

    /// A ball of solid.
    pub fn ball(radius: f32) -> Self {
        Self::new(Shape::sphere(radius))
    }

    /// Builder: place its centre here.
    pub fn at(mut self, position: Vec3) -> Self {
        self.pose.translation = position;
        self
    }

    /// Builder: use these parameters.
    pub fn with_params(mut self, params: SolidParams) -> Self {
        self.params = params;
        self
    }

    /// Builder: make it out of this.
    pub fn of(mut self, material: SolidMaterial) -> Self {
        self.params.material = material;
        self
    }

    /// Builder: start every point moving.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    /// Builder: start every point at this temperature, Kelvin.
    pub fn at_temperature(mut self, kelvin: f32) -> Self {
        self.temperature = Some(kelvin);
        self
    }

    /// The temperature this solid will actually be built at.
    pub fn resolved_temperature(&self) -> f32 {
        self.temperature
            .unwrap_or_else(|| self.params.material.natural_temperature())
    }
}

/// Sample a description into material points.
pub fn build_solid(desc: &SolidDesc) -> SolidSet {
    let mut solid = SolidSet::new(desc.params);
    solid.temperature_at_rest = desc.resolved_temperature();
    let mut points = Vec::new();
    fill(&desc.shape, desc.params.spacing, |local| {
        points.push(desc.pose.transform_point(local))
    });
    solid.deformation.reserve(points.len());
    solid.affine.reserve(points.len());
    solid.volume.reserve(points.len());
    solid.plastic.reserve(points.len());
    solid.damage.reserve(points.len());
    solid.enthalpy.reserve(points.len());
    for p in points {
        solid.push(p);
    }
    for v in &mut solid.particles.velocity {
        *v = desc.velocity;
    }
    solid
}
