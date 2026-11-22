//! Fluids: a particle set with a density constraint over it.
//!
//! PLAN.md §5 lists an SPH fluid as `ParticleSet · Material · Domain` — a
//! composition of primitives, not a new kind of object. So a [`FluidSet`]
//! *contains* the same [`ParticleSet`] cloth uses rather
//! than declaring its own point-mass storage, and everything already written
//! against that type works on fluid particles unchanged: gravity, prediction,
//! velocity recovery, and two-way contact against rigid bodies.
//!
//! What a fluid adds is one thing: the constraint. Cloth says "these two
//! particles are 5 cm apart"; a fluid says "the density around this particle
//! is 1000 kg/m³". Both are `C(x) = 0` solved by the same XPBD machinery, and
//! that is the whole reason PLAN.md §10 chose XPBD as the spine.
//!
//! ## What lives here and what does not
//!
//! Densities, Lagrange multipliers, neighbour lists and the SPH kernels are
//! *solver bookkeeping*, and they live in the solver's working set for the
//! same reason `ContactPoint` in `fusion-collide` keeps its multipliers out of
//! the contact geometry. This module holds only what is true about the fluid
//! whether or not anything is simulating it: where the particles are, how much
//! they weigh, and what density they want to be at.

use fusion_math::{Isometry, Vec3};

use crate::particle::{ParticleHandle, ParticleSet};
use crate::shape::Shape;

/// Reference to a body of fluid in a [`World`](crate::World).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FluidHandle(pub u32);

impl FluidHandle {
    /// The index this handle refers to.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The constitutive parameters of one fluid.
///
/// PLAN.md §12 defines a material as a constitutive law plus parameters. For
/// a position-based fluid the law is incompressibility and these are its
/// parameters — separate from [`Material`](crate::Material), which describes
/// bulk properties of a rigid body, because almost none of those fields mean
/// anything here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidParams {
    /// Density the fluid is solved towards, kg/m³.
    pub rest_density: f32,
    /// Lattice spacing the fluid was sampled at, metres. Particle mass follows
    /// from it as `rest_density * spacing³`, so this is the single dial that
    /// trades resolution against cost — halving it multiplies the particle
    /// count by eight.
    pub spacing: f32,
    /// Kernel support radius, metres. Every SPH sum runs over the particles
    /// within it.
    ///
    /// The ratio to `spacing` is what matters, and it is a genuine trade: at
    /// 2.0 a particle sees roughly 30 neighbours, which is the usual minimum
    /// for a density estimate that is not visibly lumpy; at 3.0 it sees about
    /// 110 and costs nearly four times as much per step.
    pub smoothing_radius: f32,
    /// Constraint-force-mixing relaxation, added to the denominator of the
    /// multiplier.
    ///
    /// Not a physical quantity — it is there because the gradient sum goes to
    /// zero for an isolated particle with no neighbours, and dividing by it
    /// would send that particle to infinity. Scaled by `rest_density²` on use
    /// so the tuned value does not change when the fluid does.
    pub relaxation: f32,
    /// XSPH viscosity coefficient, dimensionless. Blends each particle's
    /// velocity towards the average of its neighbours: 0 is inviscid and
    /// noisy, 0.1 is water, 0.5 is honey-ish. Not a physical viscosity — the
    /// effective one depends on resolution, which is worth knowing before
    /// quoting a number from it.
    pub viscosity: f32,
    /// Vorticity confinement gain. Puts back the rotational detail the
    /// density solve and XSPH damp out. Zero is off.
    pub vorticity: f32,
    /// Artificial-pressure strength — the tensile-instability fix from
    /// Macklin & Müller.
    ///
    /// **Zero by default, and that is a pairing rather than a preference.**
    /// The published method lets the density constraint act in both
    /// directions, so a surface particle is pulled inward by its own density
    /// deficit; left alone that collapses neighbours into strings, and the
    /// artificial pressure is the repulsion that stops it. Fusion instead
    /// clamps the constraint to compression only (see `solve_density`), which
    /// removes the inward pull and therefore removes the thing the repulsion
    /// was cancelling. Enabling both leaves a free surface with a net outward
    /// push and nothing opposing it: a block of water floating in vacuum
    /// expands at 8 cm/s, forever.
    ///
    /// Left as a dial because the two choices are a matched pair, and someone
    /// wanting the published behaviour needs this half of it.
    pub surface_tension: f32,
    /// Exponent on the artificial pressure term. 4 is the published value.
    pub tension_exponent: f32,
    /// Reference distance for the artificial pressure term, as a fraction of
    /// the smoothing radius. Published range is 0.1–0.3.
    pub tension_reference: f32,
    /// Dynamic viscosity, Pa·s: water is 1.0e-3, honey about 10. Read by the
    /// Unified continuum, which solves a real Newtonian viscous stress; the
    /// classic solver keeps its dimensionless [`FluidParams::viscosity`] dial.
    /// Absent, the Unified continuum maps that dial onto a viscosity instead — a
    /// documented legacy curve, so a scene saved before this existed means
    /// what it meant.
    pub dynamic_viscosity: Option<f32>,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self::WATER
    }
}

impl FluidParams {
    /// Water at 10 cm resolution — coarse enough to run interactively, fine
    /// enough that a dam break looks like a dam break.
    pub const WATER: Self = Self {
        rest_density: 1000.0,
        spacing: 0.1,
        smoothing_radius: 0.2,
        relaxation: 1.0e-4,
        viscosity: 0.05,
        vorticity: 0.1,
        surface_tension: 0.0,
        tension_exponent: 4.0,
        tension_reference: 0.2,
        // Unset, so the viscosity dial is what a scene turns, on either
        // solver; a value in Pa·s, once set, is what the Unified continuum uses.
        dynamic_viscosity: None,
    };

    /// Water, resampled at a given lattice spacing. The smoothing radius
    /// follows the spacing rather than staying put — holding it fixed while
    /// refining is the classic way to make a "convergence study" that only
    /// measures the neighbour count going up.
    pub fn water_at(spacing: f32) -> Self {
        Self {
            spacing,
            smoothing_radius: 2.0 * spacing,
            ..Self::WATER
        }
    }

    /// Builder: rest density, kg/m^3. Calibrated against the fill lattice
    /// rather than taken from a table — see the module docs for why.
    pub fn with_rest_density(mut self, density: f32) -> Self {
        self.rest_density = density;
        self
    }

    /// Builder: XSPH viscosity, dimensionless. Higher is thicker.
    pub fn with_viscosity(mut self, viscosity: f32) -> Self {
        self.viscosity = viscosity;
        self
    }

    /// Builder: dynamic viscosity, Pa·s, for the Unified continuum.
    pub fn with_dynamic_viscosity(mut self, pascal_seconds: f32) -> Self {
        self.dynamic_viscosity = Some(pascal_seconds.max(0.0));
        self
    }

    /// Builder: vorticity confinement, which puts back the swirl the density
    /// solve smooths away. It adds energy by construction.
    pub fn with_vorticity(mut self, vorticity: f32) -> Self {
        self.vorticity = vorticity;
        self
    }

    /// Mass of one particle, kg. A lattice at `spacing` fills space with one
    /// particle per `spacing³` of volume, so this is the mass that makes the
    /// lattice weigh what the fluid weighs.
    #[inline]
    pub fn particle_mass(&self) -> f32 {
        self.rest_density * self.spacing * self.spacing * self.spacing
    }

    /// Collision radius for one particle. Half the spacing: neighbouring
    /// lattice particles touch without overlapping, and the fluid presents a
    /// surface to rigid bodies at roughly the right place.
    #[inline]
    pub fn particle_radius(&self) -> f32 {
        0.5 * self.spacing
    }
}

/// A body of fluid: point masses plus the density they are solved towards.
#[derive(Clone, Debug)]
pub struct FluidSet {
    /// The point masses. Deliberately the same type cloth uses (PLAN.md §5).
    pub particles: ParticleSet,
    /// The parameters this body of fluid was built with.
    pub params: FluidParams,
    /// Each particle's affine velocity field, for the unified continuum
    /// solver, which carries a fluid as material points. Empty until that
    /// solver first steps it; the classic solver never reads it.
    pub affine: Vec<fusion_math::Mat3>,
    /// Each particle's volume over its rest volume, for the same solver: its
    /// pressure comes from this. Empty until then.
    pub volume_ratio: Vec<f32>,
}

impl FluidSet {
    /// An empty body of fluid with these parameters.
    pub fn new(params: FluidParams) -> Self {
        Self {
            particles: ParticleSet::new(),
            params,
            affine: Vec::new(),
            volume_ratio: Vec::new(),
        }
    }

    /// How many particles it holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// True where it holds none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Add one particle at the fluid's own mass and radius.
    pub fn push(&mut self, position: Vec3) -> ParticleHandle {
        // Once the Fusion APIC state exists, append history with each new
        // point. Do not reset older particles when an emitter grows a fluid.
        let tracked = !self.is_empty()
            && self.affine.len() == self.len()
            && self.volume_ratio.len() == self.len();
        let handle = self.particles.push(
            position,
            self.params.particle_mass(),
            self.params.particle_radius(),
        );
        if tracked {
            self.affine.push(fusion_math::Mat3::ZERO);
            self.volume_ratio.push(1.0);
        }
        handle
    }

    /// Total mass, kg. Reported by the diagnostics: PBF conserves mass
    /// exactly — no particle is created or destroyed — which makes this a
    /// constant, and a constant is worth asserting on.
    pub fn mass(&self) -> f32 {
        self.params.particle_mass() * self.len() as f32
    }
}

/// A body of fluid, described rather than placed.
///
/// PLAN.md §16: nobody positions a hundred thousand particles by hand. A
/// fluid is authored as a shape to fill, exactly as cloth is authored as a
/// sheet to build.
#[derive(Clone, Debug)]
pub struct FluidDesc {
    /// Region to fill. Sampled through its SDF, so any shape Fusion can
    /// describe can be filled without a special case.
    pub shape: Shape,
    /// Where the volume of fluid is, and how it is oriented.
    pub pose: Isometry,
    /// Rest density, viscosity, spacing and the rest.
    pub params: FluidParams,
    /// Velocity every particle starts with.
    pub velocity: Vec3,
}

impl FluidDesc {
    /// Fluid filling the given shape.
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            pose: Isometry::IDENTITY,
            params: FluidParams::default(),
            velocity: Vec3::ZERO,
        }
    }

    /// A rectangular volume of fluid, from its full size.
    pub fn block(size: Vec3) -> Self {
        Self::new(Shape::cuboid(size))
    }

    /// Fluid filling a sphere of `radius`.
    pub fn sphere(radius: f32) -> Self {
        Self::new(Shape::sphere(radius))
    }

    /// Builder: place its centre here.
    pub fn at(mut self, position: Vec3) -> Self {
        self.pose.translation = position;
        self
    }

    /// Builder: use these parameters.
    pub fn with_params(mut self, params: FluidParams) -> Self {
        self.params = params;
        self
    }

    /// Resample at a different lattice spacing, adjusting the smoothing radius
    /// with it.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.params = FluidParams {
            spacing,
            smoothing_radius: 2.0 * spacing,
            ..self.params
        };
        self
    }

    /// Builder: give every particle this initial velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    /// How many particles this description will produce. Cheap enough to call
    /// before committing to it, which is the point — a 5 cm dam break is
    /// 40 000 particles and worth knowing about in advance.
    pub fn particle_count(&self) -> usize {
        let mut n = 0;
        fill(&self.shape, self.params.spacing, |_| n += 1);
        n
    }
}

/// Sample the interior of a shape on a lattice, in the shape's own frame.
///
/// The one authoring verb of PLAN.md §16. Points are emitted in a fixed
/// x-fastest order so that a fill is reproducible — the particle at index 400
/// is the same particle on every run, which replay (§17) depends on.
///
/// The lattice is the one whose cells tile the shape's bounds, so a box's
/// outermost particles sit half a spacing in and the cells they stand for
/// tile the box exactly. A particle is kept where its centre is inside the
/// shape: each one stands for its own cell, and on a curved surface the cells
/// that poke out and the corners left uncovered even out.
///
/// It used to keep a particle only where its centre was half a spacing
/// inside, which leaves a box alone and takes a half-spacing shell off
/// everything curved: a 1 m deformable ball at the editor's default sampling
/// held 74% of its volume and mass and was 0.75 m across, so it shrank the
/// moment it was played.
pub fn fill(shape: &Shape, spacing: f32, mut emit: impl FnMut(Vec3)) {
    let spacing = spacing.max(1e-6);
    let bounds = shape.local_aabb();
    let inset = 0.5 * spacing;
    // Both decisions below are "is this exactly on the boundary", and both are
    // asked in floating point about numbers a user chose to be exact.
    //
    // A 40 cm box at 10 cm resolution is four points across, and it was two:
    // `0.4 - 0.1` is `0.30000001`, dividing by `0.1` gives `3.0000002`, the
    // floor is 3 rather than 3, and then the outermost ring lands at exactly
    // one inset from the surface where the comparison decides by the last bit.
    // The result was a solid **half the size the scene asked for**, holding an
    // eighth of the material, with nothing anywhere reporting a problem — and
    // whether it happened at all depended on whether the division rounded up
    // or down, so 40 cm at 5 cm was right and 40 cm at 10 cm was not.
    //
    // A ten-thousandth of a spacing is far below anything a lattice resolves
    // and far above the rounding.
    let slop = 1e-4 * spacing;

    // Centre the lattice on the shape rather than starting at its minimum, so
    // that a symmetric shape gets a symmetric fill. Starting at the minimum
    // leaves whatever the division rounded off as a one-sided gap, and a block
    // of water then sits visibly off-centre in its container.
    let extents = bounds.extents();
    let counts = [0, 1, 2].map(|k| {
        let usable = extents[k] - 2.0 * inset;
        if usable < 0.0 {
            0
        } else {
            (usable / spacing + 1e-4).floor() as i32 + 1
        }
    });
    if counts.iter().any(|&c| c <= 0) {
        return;
    }
    let centre = bounds.center();
    let origin = Vec3::new(
        centre.x - 0.5 * (counts[0] - 1) as f32 * spacing,
        centre.y - 0.5 * (counts[1] - 1) as f32 * spacing,
        centre.z - 0.5 * (counts[2] - 1) as f32 * spacing,
    );

    for iz in 0..counts[2] {
        for iy in 0..counts[1] {
            for ix in 0..counts[0] {
                let p = origin
                    + Vec3::new(
                        ix as f32 * spacing,
                        iy as f32 * spacing,
                        iz as f32 * spacing,
                    );
                if shape.signed_distance_local(p) <= slop {
                    emit(p);
                }
            }
        }
    }
}

/// Build a fluid from its description.
pub fn build_fluid(desc: &FluidDesc) -> FluidSet {
    let mut set = FluidSet::new(desc.params);
    let pose = desc.pose;
    fill(&desc.shape, desc.params.spacing, |local| {
        set.push(pose.transform_point(local));
    });
    for v in &mut set.particles.velocity {
        *v = desc.velocity;
    }
    set
}
