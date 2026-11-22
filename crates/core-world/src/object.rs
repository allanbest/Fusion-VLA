//! Things in the world: a material, a representation, and its state.

use core_math::{Mat3, Pose, Vec3};
use core_space::Shape;
use std::sync::Arc;

use crate::columns::Columns;
use crate::mesh::{Surface, TetMesh};
use crate::MaterialId;

/// An object.
#[derive(Clone, Debug)]
pub struct Object {
    pub name: String,
    pub material: MaterialId,
    pub representation: Representation,
}

/// How an object's state is stored — the one place discretisation lives.
/// More join as modules need them: reduced solids, particles, material
/// points, point masses.
#[derive(Clone, Debug)]
pub enum Representation {
    /// Does not move: the ground, walls, a fixed support.
    Fixed(Fixed),
    /// Moves, does not deform.
    Rigid(Rigid),
    /// Deforms: finite elements on a tetrahedral mesh.
    Solid(Solid),
}

#[derive(Clone, Debug)]
pub struct Fixed {
    pub shape: Shape,
    pub pose: Pose,
}

/// A rigid body: six degrees of freedom.
#[derive(Clone, Debug)]
pub struct Rigid {
    pub shape: Shape,
    /// Of the centre of mass and the body's axes.
    pub pose: Pose,
    /// Of the centre of mass, m/s.
    pub velocity: Vec3,
    /// Angular velocity in the world frame, rad/s.
    pub spin: Vec3,
    /// kg.
    pub mass: f64,
    /// About the centre of mass, in the body's frame, kg m².
    pub inertia: Mat3,
}

impl Rigid {
    /// Inertia about the centre of mass in the world frame.
    pub fn world_inertia(&self) -> Mat3 {
        let r = self.pose.rotation.to_mat3();
        r * self.inertia * r.transpose()
    }

    pub fn angular_momentum(&self) -> Vec3 {
        self.world_inertia() * self.spin
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.length_squared()
            + 0.5 * self.spin.dot(self.angular_momentum())
    }

    /// Velocity of the body's material point at world position `p`.
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(p - self.pose.position)
    }
}

/// A deformable solid: a node per mesh vertex.
#[derive(Clone, Debug)]
pub struct Solid {
    /// Immutable geometry shared by compact parallel simulation views.
    pub mesh: Arc<TetMesh>,
    pub surface: Arc<Surface>,
    pub nodes: Nodes,
    /// Per-tetrahedron attributes (damage, plastic strain…).
    pub elements: Columns,
}

/// Node state, as columns.
#[derive(Clone, Debug, Default)]
pub struct Nodes {
    /// m.
    pub position: Vec<Vec3>,
    /// m/s.
    pub velocity: Vec<Vec3>,
    /// Lumped mass, kg.
    pub mass: Vec<f64>,
    /// Per-node attributes (temperature…).
    pub columns: Columns,
}

impl Nodes {
    pub fn len(&self) -> usize {
        self.position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }
}

impl Solid {
    /// The mesh placed at `pose`, at rest, with `density` lumped onto its
    /// nodes ([`TetMesh::mass_shares`]).
    pub fn new(mesh: TetMesh, pose: &Pose, density: f64) -> Self {
        assert!(
            mesh.is_valid(),
            "a tetrahedron is inverted, flat, or indexes a missing node"
        );
        let n = mesh.rest.len();
        let mut mass = vec![0.0; n];
        let shares = mesh.mass_shares();
        for (e, v) in mesh.volumes().into_iter().enumerate() {
            for (&node, share) in mesh.element(e).iter().zip(&shares) {
                mass[node as usize] += density * v * share;
            }
        }
        let nodes = Nodes {
            position: mesh.rest.iter().map(|&p| pose.transform_point(p)).collect(),
            velocity: vec![Vec3::ZERO; n],
            mass,
            columns: Columns::new(n),
        };
        let elements = Columns::new(mesh.tets.len());
        Self {
            surface: Arc::new(mesh.surface()),
            mesh: Arc::new(mesh),
            nodes,
            elements,
        }
    }
}

impl Object {
    /// kg; infinite for a fixed object.
    pub fn mass(&self) -> f64 {
        match &self.representation {
            Representation::Fixed(_) => f64::INFINITY,
            Representation::Rigid(r) => r.mass,
            Representation::Solid(s) => s.nodes.mass.iter().sum(),
        }
    }

    /// Whether it moves at all.
    pub fn is_dynamic(&self) -> bool {
        !matches!(self.representation, Representation::Fixed(_))
    }

    /// m; `None` for a fixed object.
    pub fn centre_of_mass(&self) -> Option<Vec3> {
        match &self.representation {
            Representation::Fixed(_) => None,
            Representation::Rigid(r) => Some(r.pose.position),
            Representation::Solid(s) => {
                let n = &s.nodes;
                Some(
                    n.position
                        .iter()
                        .zip(&n.mass)
                        .map(|(&p, &m)| p * m)
                        .sum::<Vec3>()
                        / self.mass(),
                )
            }
        }
    }

    /// Linear momentum, kg m/s.
    pub fn momentum(&self) -> Vec3 {
        match &self.representation {
            Representation::Fixed(_) => Vec3::ZERO,
            Representation::Rigid(r) => r.velocity * r.mass,
            Representation::Solid(s) => s
                .nodes
                .velocity
                .iter()
                .zip(&s.nodes.mass)
                .map(|(&v, &m)| v * m)
                .sum(),
        }
    }

    /// Angular momentum about the world origin, kg m²/s.
    pub fn angular_momentum(&self) -> Vec3 {
        match &self.representation {
            Representation::Fixed(_) => Vec3::ZERO,
            Representation::Rigid(r) => {
                r.pose.position.cross(r.velocity * r.mass) + r.angular_momentum()
            }
            Representation::Solid(s) => {
                let n = &s.nodes;
                (0..n.len())
                    .map(|i| n.position[i].cross(n.velocity[i] * n.mass[i]))
                    .sum()
            }
        }
    }

    /// J.
    pub fn kinetic_energy(&self) -> f64 {
        match &self.representation {
            Representation::Fixed(_) => 0.0,
            Representation::Rigid(r) => r.kinetic_energy(),
            Representation::Solid(s) => s
                .nodes
                .velocity
                .iter()
                .zip(&s.nodes.mass)
                .map(|(v, m)| 0.5 * m * v.length_squared())
                .sum(),
        }
    }

    /// Potential energy in a uniform field `gravity`, relative to the
    /// origin, J.
    pub fn gravitational_energy(&self, gravity: Vec3) -> f64 {
        match &self.representation {
            Representation::Fixed(_) => 0.0,
            Representation::Rigid(r) => -r.mass * gravity.dot(r.pose.position),
            Representation::Solid(s) => s
                .nodes
                .position
                .iter()
                .zip(&s.nodes.mass)
                .map(|(p, m)| -m * gravity.dot(*p))
                .sum(),
        }
    }
}
