//! `fusion_vla.core` — Python bindings for the `f64` variational Fusion Core
//! solver, URDF/MJCF robotics importers, vectorized RL environments
//! (`WorldBatch` + `DomainRandomizer`), virtual instrumentation (`Probe`),
//! and differentiable Implicit-Function-Theorem adjoints (`backward_step`).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use core_math::{Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{
    library, JointDrive, Link, LinkId, LinkPoint, Material, MaterialId, ObjectId, Probe, ProbeId,
    ProbeReading, Representation, RobotModel, TetMesh, World,
};
use phys_mechanics::{
    AdjointGradients, AdjointLoss, DomainRandomizer, Integrator, Mechanics, MechanicsState,
    Settings, StepReport, WorldBatch,
};
use pyo3::exceptions::{PyIOError, PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Vec3Arg = (f64, f64, f64);
type QuatArg = (f64, f64, f64, f64);

fn to_vec3(v: Vec3Arg) -> PyResult<Vec3> {
    if !v.0.is_finite() || !v.1.is_finite() || !v.2.is_finite() {
        return Err(PyValueError::new_err("vector components must be finite"));
    }
    Ok(Vec3::new(v.0, v.1, v.2))
}

fn from_vec3(v: Vec3) -> Vec3Arg {
    (v.x, v.y, v.z)
}

fn to_quat(q: QuatArg) -> PyResult<Quat> {
    if !q.0.is_finite() || !q.1.is_finite() || !q.2.is_finite() || !q.3.is_finite() {
        return Err(PyValueError::new_err(
            "quaternion components must be finite",
        ));
    }
    let len2 = q.0 * q.0 + q.1 * q.1 + q.2 * q.2 + q.3 * q.3;
    if len2 < 1e-18 {
        return Err(PyValueError::new_err("quaternion cannot have zero norm"));
    }
    Ok(Quat::from_xyzw(q.0, q.1, q.2, q.3).normalize())
}

fn from_quat(q: Quat) -> QuatArg {
    (q.x, q.y, q.z, q.w)
}

fn to_pose(pos: Vec3Arg, quat: QuatArg) -> PyResult<Pose> {
    Ok(Pose::new(to_vec3(pos)?, to_quat(quat)?))
}

fn check_pos(v: f64, name: &str) -> PyResult<f64> {
    if !v.is_finite() || v <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(v)
}

fn check_nonneg(v: f64, name: &str) -> PyResult<f64> {
    if !v.is_finite() || v < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(v)
}

fn check_material_id(world: &World, id: MaterialId) -> PyResult<MaterialId> {
    if (id.0 as usize) >= world.materials().len() {
        return Err(PyIndexError::new_err(format!(
            "MaterialId({}) out of range",
            id.0
        )));
    }
    Ok(id)
}

fn check_object_id(world: &World, id: ObjectId) -> PyResult<ObjectId> {
    if (id.0 as usize) >= world.objects().len() {
        return Err(PyIndexError::new_err(format!(
            "ObjectId({}) out of range",
            id.0
        )));
    }
    Ok(id)
}

fn check_rigid_or_fixed(world: &World, id: ObjectId) -> PyResult<ObjectId> {
    check_object_id(world, id)?;
    if matches!(world.object(id).representation, Representation::Solid(_)) {
        return Err(PyValueError::new_err(format!(
            "ObjectId({}) is a deformable solid; expected rigid or fixed",
            id.0
        )));
    }
    Ok(id)
}

fn check_link_id(world: &World, id: LinkId) -> PyResult<LinkId> {
    if (id.0 as usize) >= world.links().len() {
        return Err(PyIndexError::new_err(format!(
            "LinkId({}) out of range",
            id.0
        )));
    }
    Ok(id)
}

fn check_link_point(world: &World, pt: LinkPoint) -> PyResult<LinkPoint> {
    match pt {
        LinkPoint::World(p) => {
            if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                return Err(PyValueError::new_err("LinkPoint::World must be finite"));
            }
        }
        LinkPoint::Body { object, local } => {
            check_rigid_or_fixed(world, object)?;
            if !local.x.is_finite() || !local.y.is_finite() || !local.z.is_finite() {
                return Err(PyValueError::new_err(
                    "LinkPoint::Body local must be finite",
                ));
            }
        }
        LinkPoint::Node { object, node } => {
            check_object_id(world, object)?;
            let Representation::Solid(s) = &world.object(object).representation else {
                return Err(PyValueError::new_err(
                    "LinkPoint::Node requires a deformable solid object",
                ));
            };
            if (node as usize) >= s.nodes.len() {
                return Err(PyIndexError::new_err(format!(
                    "solid node {node} out of range (len {})",
                    s.nodes.len()
                )));
            }
        }
    }
    Ok(pt)
}

fn parse_integrator(name: &str) -> PyResult<Integrator> {
    match name.to_ascii_lowercase().as_str() {
        "tr_bdf2" | "trbdf2" => Ok(Integrator::TrBdf2),
        "bdf2" => Ok(Integrator::Bdf2),
        "backward_euler" | "euler" => Ok(Integrator::BackwardEuler),
        "newmark" => Ok(Integrator::Newmark),
        other => Err(PyValueError::new_err(format!(
            "unknown integrator '{other}'; expected tr_bdf2, bdf2, backward_euler, or newmark"
        ))),
    }
}

fn integrator_name(integrator: Integrator) -> &'static str {
    match integrator {
        Integrator::TrBdf2 => "tr_bdf2",
        Integrator::Bdf2 => "bdf2",
        Integrator::BackwardEuler => "backward_euler",
        Integrator::Newmark => "newmark",
    }
}

macro_rules! id_handle {
    ($py_name:ident, $rust_ty:ty, $py_str:literal) => {
        #[pyclass(module = "fusion_vla.core", name = $py_str)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $py_name {
            pub(crate) inner: $rust_ty,
        }

        #[pymethods]
        impl $py_name {
            #[getter]
            fn index(&self) -> u32 {
                self.inner.0
            }

            fn __eq__(&self, other: &Self) -> bool {
                self.inner == other.inner
            }

            fn __hash__(&self) -> u64 {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                self.inner.0.hash(&mut h);
                h.finish()
            }

            fn __repr__(&self) -> String {
                format!("<fusion_vla.core.{} {}>", $py_str, self.inner.0)
            }
        }
    };
}

id_handle!(PyMaterialId, MaterialId, "MaterialId");
id_handle!(PyObjectId, ObjectId, "ObjectId");
id_handle!(PyLinkId, LinkId, "LinkId");
id_handle!(PyProbeId, ProbeId, "ProbeId");

#[pyclass(module = "fusion_vla.core", name = "LinkPoint")]
#[derive(Clone, Copy)]
pub struct PyCoreLinkPoint {
    pub(crate) inner: LinkPoint,
}

#[pymethods]
impl PyCoreLinkPoint {
    #[staticmethod]
    fn world(pos: Vec3Arg) -> PyResult<Self> {
        Ok(Self {
            inner: LinkPoint::World(to_vec3(pos)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (object, local = (0.0, 0.0, 0.0)))]
    fn body(object: PyObjectId, local: Vec3Arg) -> PyResult<Self> {
        Ok(Self {
            inner: LinkPoint::Body {
                object: object.inner,
                local: to_vec3(local)?,
            },
        })
    }

    #[staticmethod]
    fn node(object: PyObjectId, node: u32) -> Self {
        Self {
            inner: LinkPoint::Node {
                object: object.inner,
                node,
            },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            LinkPoint::World(_) => "world",
            LinkPoint::Body { .. } => "body",
            LinkPoint::Node { .. } => "node",
        }
    }

    #[getter]
    fn object(&self) -> Option<PyObjectId> {
        self.inner.object().map(|inner| PyObjectId { inner })
    }

    fn __repr__(&self) -> String {
        match self.inner {
            LinkPoint::World(p) => {
                format!("<LinkPoint world ({:.3}, {:.3}, {:.3})>", p.x, p.y, p.z)
            }
            LinkPoint::Body { object, local } => format!(
                "<LinkPoint body={} local=({:.3}, {:.3}, {:.3})>",
                object.0, local.x, local.y, local.z
            ),
            LinkPoint::Node { object, node } => {
                format!("<LinkPoint solid={} node={}>", object.0, node)
            }
        }
    }
}

#[pyclass(module = "fusion_vla.core", name = "Material")]
#[derive(Clone)]
pub struct PyCoreMaterial {
    inner: Material,
}

#[pymethods]
impl PyCoreMaterial {
    #[new]
    #[pyo3(signature = (name, density, youngs, poisson, friction = 0.5, restitution = 0.2))]
    fn new(
        name: &str,
        density: f64,
        youngs: f64,
        poisson: f64,
        friction: f64,
        restitution: f64,
    ) -> PyResult<Self> {
        check_pos(density, "density")?;
        check_nonneg(youngs, "youngs")?;
        if !poisson.is_finite() || !(0.0..0.5).contains(&poisson) {
            return Err(PyValueError::new_err("poisson must be in [0, 0.5)"));
        }
        check_nonneg(friction, "friction")?;
        if !restitution.is_finite() || !(0.0..=1.0).contains(&restitution) {
            return Err(PyValueError::new_err("restitution must be in [0, 1]"));
        }
        Ok(Self {
            inner: Material::new(name, density, youngs, poisson, friction, restitution),
        })
    }

    #[staticmethod]
    fn steel() -> Self {
        Self {
            inner: library::structural_steel(),
        }
    }

    #[staticmethod]
    fn aluminium() -> Self {
        Self {
            inner: library::aluminium(),
        }
    }

    #[staticmethod]
    fn concrete() -> Self {
        Self {
            inner: library::concrete(),
        }
    }

    #[staticmethod]
    fn glass() -> Self {
        Self {
            inner: library::glass(),
        }
    }

    #[staticmethod]
    fn granite() -> Self {
        Self {
            inner: library::granite(),
        }
    }

    #[staticmethod]
    fn oak() -> Self {
        Self {
            inner: library::oak(),
        }
    }

    #[staticmethod]
    fn ice() -> Self {
        Self {
            inner: library::ice(),
        }
    }

    #[staticmethod]
    fn rubber() -> Self {
        Self {
            inner: library::rubber(),
        }
    }

    #[staticmethod]
    fn gelatin() -> Self {
        Self {
            inner: library::gelatin(),
        }
    }

    #[pyo3(signature = (yield_stress, hardening = 0.0))]
    fn with_plasticity(&self, yield_stress: f64, hardening: f64) -> PyResult<Self> {
        check_pos(yield_stress, "yield_stress")?;
        check_nonneg(hardening, "hardening")?;
        Ok(Self {
            inner: self.inner.clone().with_plasticity(yield_stress, hardening),
        })
    }

    #[pyo3(signature = (toughness, softening, residual_stiffness = 1e-3))]
    fn with_fracture(
        &self,
        toughness: f64,
        softening: f64,
        residual_stiffness: f64,
    ) -> PyResult<Self> {
        check_pos(toughness, "toughness")?;
        check_pos(softening, "softening")?;
        check_pos(residual_stiffness, "residual_stiffness")?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_fracture(toughness, softening, residual_stiffness),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn density(&self) -> f64 {
        self.inner.density
    }

    #[getter]
    fn youngs(&self) -> f64 {
        self.inner.youngs.nominal()
    }

    #[getter]
    fn poisson(&self) -> f64 {
        self.inner.poisson.nominal()
    }

    #[getter]
    fn friction(&self) -> f64 {
        self.inner.friction
    }

    #[setter]
    fn set_friction(&mut self, value: f64) -> PyResult<()> {
        self.inner.friction = check_nonneg(value, "friction")?;
        Ok(())
    }

    #[getter]
    fn restitution(&self) -> f64 {
        self.inner.restitution
    }

    #[setter]
    fn set_restitution(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(PyValueError::new_err("restitution must be in [0, 1]"));
        }
        self.inner.restitution = value;
        Ok(())
    }
}

#[pyclass(module = "fusion_vla.core", name = "JointDrive")]
#[derive(Clone, Copy)]
pub struct PyCoreJointDrive {
    pub(crate) inner: JointDrive,
}

#[pymethods]
impl PyCoreJointDrive {
    #[new]
    #[pyo3(signature = (target = 0.0, stiffness = 0.0, damping = 0.0, target_velocity = 0.0, max_force = None))]
    fn new(
        target: f64,
        stiffness: f64,
        damping: f64,
        target_velocity: f64,
        max_force: Option<f64>,
    ) -> PyResult<Self> {
        if !target.is_finite() || !target_velocity.is_finite() {
            return Err(PyValueError::new_err("drive targets must be finite"));
        }
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(damping, "damping")?;
        if let Some(mf) = max_force {
            check_pos(mf, "max_force")?;
        }
        Ok(Self {
            inner: JointDrive {
                target,
                target_velocity,
                stiffness,
                damping,
                max_force,
            },
        })
    }

    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: JointDrive::NONE,
        }
    }

    #[getter]
    fn target(&self) -> f64 {
        self.inner.target
    }
    #[setter]
    fn set_target(&mut self, val: f64) -> PyResult<()> {
        if !val.is_finite() {
            return Err(PyValueError::new_err("target must be finite"));
        }
        self.inner.target = val;
        Ok(())
    }

    #[getter]
    fn target_velocity(&self) -> f64 {
        self.inner.target_velocity
    }
    #[setter]
    fn set_target_velocity(&mut self, val: f64) -> PyResult<()> {
        if !val.is_finite() {
            return Err(PyValueError::new_err("target_velocity must be finite"));
        }
        self.inner.target_velocity = val;
        Ok(())
    }

    #[getter]
    fn stiffness(&self) -> f64 {
        self.inner.stiffness
    }
    #[setter]
    fn set_stiffness(&mut self, val: f64) -> PyResult<()> {
        self.inner.stiffness = check_nonneg(val, "stiffness")?;
        Ok(())
    }

    #[getter]
    fn damping(&self) -> f64 {
        self.inner.damping
    }
    #[setter]
    fn set_damping(&mut self, val: f64) -> PyResult<()> {
        self.inner.damping = check_nonneg(val, "damping")?;
        Ok(())
    }

    #[getter]
    fn max_force(&self) -> Option<f64> {
        self.inner.max_force
    }
    #[setter]
    fn set_max_force(&mut self, val: Option<f64>) -> PyResult<()> {
        if let Some(mf) = val {
            check_pos(mf, "max_force")?;
        }
        self.inner.max_force = val;
        Ok(())
    }
}

#[pyclass(module = "fusion_vla.core", name = "RobotModel")]
#[derive(Clone)]
pub struct PyCoreRobotModel {
    inner: RobotModel,
}

#[pymethods]
impl PyCoreRobotModel {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn root(&self) -> PyObjectId {
        PyObjectId {
            inner: self.inner.root,
        }
    }

    fn link(&self, name: &str) -> PyResult<PyObjectId> {
        self.inner
            .link(name)
            .map(|inner| PyObjectId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown link '{name}'")))
    }

    fn joint(&self, name: &str) -> PyResult<PyLinkId> {
        self.inner
            .joint(name)
            .map(|inner| PyLinkId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown joint '{name}'")))
    }

    fn actuator(&self, name: &str) -> PyResult<PyLinkId> {
        self.inner
            .actuator(name)
            .map(|inner| PyLinkId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown actuator '{name}'")))
    }

    fn tendon(&self, name: &str) -> PyResult<PyLinkId> {
        self.inner
            .tendon(name)
            .map(|inner| PyLinkId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown tendon '{name}'")))
    }

    fn site(&self, name: &str) -> PyResult<PyCoreLinkPoint> {
        self.inner
            .site(name)
            .map(|inner| PyCoreLinkPoint { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown site '{name}'")))
    }

    fn probe(&self, name: &str) -> PyResult<PyProbeId> {
        self.inner
            .probe(name)
            .map(|inner| PyProbeId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown probe '{name}'")))
    }

    fn geom(&self, name: &str) -> PyResult<PyObjectId> {
        self.inner
            .geom(name)
            .map(|inner| PyObjectId { inner })
            .ok_or_else(|| PyKeyError::new_err(format!("unknown geom '{name}'")))
    }

    fn link_names(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.inner.links.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn joint_names(&self) -> Vec<String> {
        self.inner.joint_names()
    }

    fn actuator_names(&self) -> Vec<String> {
        self.inner.actuator_names()
    }

    fn tendon_names(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.inner.tendons.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn site_names(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.inner.sites.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn probe_names(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.inner.probes.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn geom_names(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.inner.geoms.keys().cloned().collect();
        keys.sort();
        keys
    }

    #[getter]
    fn links(&self) -> HashMap<String, PyObjectId> {
        self.inner
            .links
            .iter()
            .map(|(k, &v)| (k.clone(), PyObjectId { inner: v }))
            .collect()
    }

    #[getter]
    fn joints(&self) -> HashMap<String, PyLinkId> {
        self.inner
            .joints
            .iter()
            .map(|(k, &v)| (k.clone(), PyLinkId { inner: v }))
            .collect()
    }

    #[getter]
    fn actuators(&self) -> HashMap<String, PyLinkId> {
        self.inner
            .actuators
            .iter()
            .map(|(k, &v)| (k.clone(), PyLinkId { inner: v }))
            .collect()
    }

    #[getter]
    fn tendons(&self) -> HashMap<String, PyLinkId> {
        self.inner
            .tendons
            .iter()
            .map(|(k, &v)| (k.clone(), PyLinkId { inner: v }))
            .collect()
    }

    #[getter]
    fn probes(&self) -> HashMap<String, PyProbeId> {
        self.inner
            .probes
            .iter()
            .map(|(k, &v)| (k.clone(), PyProbeId { inner: v }))
            .collect()
    }

    #[getter]
    fn geoms(&self) -> HashMap<String, PyObjectId> {
        self.inner
            .geoms
            .iter()
            .map(|(k, &v)| (k.clone(), PyObjectId { inner: v }))
            .collect()
    }
}

#[pyclass(module = "fusion_vla.core", name = "World")]
#[derive(Clone)]
pub struct PyCoreWorld {
    pub(crate) inner: World,
}

#[pymethods]
impl PyCoreWorld {
    #[new]
    #[pyo3(signature = (gravity = (0.0, -9.81, 0.0)))]
    fn new(gravity: Vec3Arg) -> PyResult<Self> {
        let mut world = World::new();
        world.gravity = to_vec3(gravity)?;
        Ok(Self { inner: world })
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    #[getter]
    fn gravity(&self) -> Vec3Arg {
        from_vec3(self.inner.gravity)
    }
    #[setter]
    fn set_gravity(&mut self, gravity: Vec3Arg) -> PyResult<()> {
        self.inner.gravity = to_vec3(gravity)?;
        Ok(())
    }

    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }

    #[getter]
    fn object_count(&self) -> usize {
        self.inner.objects().len()
    }

    #[getter]
    fn link_count(&self) -> usize {
        self.inner.links().len()
    }

    #[getter]
    fn probe_count(&self) -> usize {
        self.inner.probes().len()
    }

    fn add_material(&mut self, material: &PyCoreMaterial) -> PyMaterialId {
        PyMaterialId {
            inner: self.inner.add_material(material.inner.clone()),
        }
    }

    #[pyo3(signature = (material, y = 0.0, name = "ground"))]
    fn add_ground(&mut self, material: PyMaterialId, y: f64, name: &str) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        if !y.is_finite() {
            return Err(PyValueError::new_err("ground y must be finite"));
        }
        Ok(PyObjectId {
            inner: self
                .inner
                .add_fixed(name, material.inner, Shape::ground(y), Pose::IDENTITY),
        })
    }

    #[pyo3(signature = (material, normal = (0.0, 1.0, 0.0), offset = 0.0, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "plane"))]
    #[allow(clippy::too_many_arguments)]
    fn add_fixed_plane(
        &mut self,
        material: PyMaterialId,
        normal: Vec3Arg,
        offset: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        let n = to_vec3(normal)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("plane normal cannot be zero"))?;
        if !offset.is_finite() {
            return Err(PyValueError::new_err("plane offset must be finite"));
        }
        Ok(PyObjectId {
            inner: self.inner.add_fixed(
                name,
                material.inner,
                Shape::HalfSpace { normal: n, offset },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, half_extents, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "fixed_box"))]
    fn add_fixed_box(
        &mut self,
        material: PyMaterialId,
        half_extents: Vec3Arg,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        let h = to_vec3(half_extents)?;
        check_pos(h.x, "half_extents.x")?;
        check_pos(h.y, "half_extents.y")?;
        check_pos(h.z, "half_extents.z")?;
        Ok(PyObjectId {
            inner: self.inner.add_fixed(
                name,
                material.inner,
                Shape::Cuboid { half: h },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, radius, pos = (0.0, 0.0, 0.0), name = "fixed_sphere"))]
    fn add_fixed_sphere(
        &mut self,
        material: PyMaterialId,
        radius: f64,
        pos: Vec3Arg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        check_pos(radius, "radius")?;
        Ok(PyObjectId {
            inner: self.inner.add_fixed(
                name,
                material.inner,
                Shape::Sphere { radius },
                Pose::at(to_vec3(pos)?),
            ),
        })
    }

    #[pyo3(signature = (material, radius, half_length, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "fixed_capsule"))]
    fn add_fixed_capsule(
        &mut self,
        material: PyMaterialId,
        radius: f64,
        half_length: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        check_pos(radius, "radius")?;
        check_nonneg(half_length, "half_length")?;
        Ok(PyObjectId {
            inner: self.inner.add_fixed(
                name,
                material.inner,
                Shape::Capsule {
                    radius,
                    half_length,
                },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, half_extents, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "box"))]
    fn add_rigid_box(
        &mut self,
        material: PyMaterialId,
        half_extents: Vec3Arg,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        let h = to_vec3(half_extents)?;
        check_pos(h.x, "half_extents.x")?;
        check_pos(h.y, "half_extents.y")?;
        check_pos(h.z, "half_extents.z")?;
        Ok(PyObjectId {
            inner: self.inner.add_rigid(
                name,
                material.inner,
                Shape::Cuboid { half: h },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, radius, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "sphere"))]
    fn add_rigid_sphere(
        &mut self,
        material: PyMaterialId,
        radius: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        check_pos(radius, "radius")?;
        Ok(PyObjectId {
            inner: self.inner.add_rigid(
                name,
                material.inner,
                Shape::Sphere { radius },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, radius, half_length, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), name = "capsule"))]
    fn add_rigid_capsule(
        &mut self,
        material: PyMaterialId,
        radius: f64,
        half_length: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        check_pos(radius, "radius")?;
        check_nonneg(half_length, "half_length")?;
        Ok(PyObjectId {
            inner: self.inner.add_rigid(
                name,
                material.inner,
                Shape::Capsule {
                    radius,
                    half_length,
                },
                to_pose(pos, quat)?,
            ),
        })
    }

    #[pyo3(signature = (material, size, edge, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), quadratic = true, name = "solid_box"))]
    #[allow(clippy::too_many_arguments)]
    fn add_solid_box(
        &mut self,
        material: PyMaterialId,
        size: Vec3Arg,
        edge: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        quadratic: bool,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        let s = to_vec3(size)?;
        check_pos(s.x, "size.x")?;
        check_pos(s.y, "size.y")?;
        check_pos(s.z, "size.z")?;
        check_pos(edge, "edge")?;
        let base = TetMesh::cuboid(s, edge);
        let mesh = if quadratic { base.quadratic() } else { base };
        Ok(PyObjectId {
            inner: self
                .inner
                .add_solid(name, material.inner, mesh, to_pose(pos, quat)?),
        })
    }

    #[pyo3(signature = (material, radius, edge, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), quadratic = true, name = "solid_ball"))]
    #[allow(clippy::too_many_arguments)]
    fn add_solid_ball(
        &mut self,
        material: PyMaterialId,
        radius: f64,
        edge: f64,
        pos: Vec3Arg,
        quat: QuatArg,
        quadratic: bool,
        name: &str,
    ) -> PyResult<PyObjectId> {
        check_material_id(&self.inner, material.inner)?;
        check_pos(radius, "radius")?;
        check_pos(edge, "edge")?;
        let base = TetMesh::ball(radius, edge);
        if base.tets.is_empty() {
            return Err(PyValueError::new_err(
                "edge too large relative to radius to generate tetrahedra",
            ));
        }
        let mesh = if quadratic { base.quadratic() } else { base };
        Ok(PyObjectId {
            inner: self
                .inner
                .add_solid(name, material.inner, mesh, to_pose(pos, quat)?),
        })
    }

    fn anchor_in_box(
        &mut self,
        object: PyObjectId,
        min_bound: Vec3Arg,
        max_bound: Vec3Arg,
    ) -> PyResult<Option<PyLinkId>> {
        check_object_id(&self.inner, object.inner)?;
        if !matches!(
            self.inner.object(object.inner).representation,
            Representation::Solid(_)
        ) {
            return Err(PyValueError::new_err(
                "anchor_in_box requires a deformable solid object",
            ));
        }
        let lo = to_vec3(min_bound)?;
        let hi = to_vec3(max_bound)?;
        Ok(self
            .inner
            .anchor(object.inner, |p| {
                p.x >= lo.x
                    && p.x <= hi.x
                    && p.y >= lo.y
                    && p.y <= hi.y
                    && p.z >= lo.z
                    && p.z <= hi.z
            })
            .map(|inner| PyLinkId { inner }))
    }

    #[pyo3(signature = (xml, material, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), fixed_base = true))]
    fn load_urdf(
        &mut self,
        xml: &str,
        material: PyMaterialId,
        pos: Vec3Arg,
        quat: QuatArg,
        fixed_base: bool,
    ) -> PyResult<PyCoreRobotModel> {
        check_material_id(&self.inner, material.inner)?;
        let pose = to_pose(pos, quat)?;
        let model = self
            .inner
            .load_urdf(xml, material.inner, pose, fixed_base)
            .map_err(PyValueError::new_err)?;
        Ok(PyCoreRobotModel { inner: model })
    }

    #[pyo3(signature = (path, material, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0), fixed_base = true))]
    fn load_urdf_file(
        &mut self,
        path: &str,
        material: PyMaterialId,
        pos: Vec3Arg,
        quat: QuatArg,
        fixed_base: bool,
    ) -> PyResult<PyCoreRobotModel> {
        let xml = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("failed to read URDF '{path}': {e}")))?;
        self.load_urdf(&xml, material, pos, quat, fixed_base)
    }

    #[pyo3(signature = (xml, material, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0)))]
    fn load_mjcf(
        &mut self,
        xml: &str,
        material: PyMaterialId,
        pos: Vec3Arg,
        quat: QuatArg,
    ) -> PyResult<PyCoreRobotModel> {
        check_material_id(&self.inner, material.inner)?;
        let pose = to_pose(pos, quat)?;
        let model = self
            .inner
            .load_mjcf(xml, material.inner, pose)
            .map_err(PyValueError::new_err)?;
        Ok(PyCoreRobotModel { inner: model })
    }

    #[pyo3(signature = (path, material, pos = (0.0, 0.0, 0.0), quat = (0.0, 0.0, 0.0, 1.0)))]
    fn load_mjcf_file(
        &mut self,
        path: &str,
        material: PyMaterialId,
        pos: Vec3Arg,
        quat: QuatArg,
    ) -> PyResult<PyCoreRobotModel> {
        let xml = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("failed to read MJCF '{path}': {e}")))?;
        self.load_mjcf(&xml, material, pos, quat)
    }

    fn exclude_contact(&mut self, a: PyObjectId, b: PyObjectId) -> PyResult<()> {
        check_object_id(&self.inner, a.inner)?;
        check_object_id(&self.inner, b.inner)?;
        self.inner.exclude_contact(a.inner, b.inner);
        Ok(())
    }

    #[pyo3(signature = (a, b, rest_length, stiffness, damping = 0.0))]
    fn add_spring(
        &mut self,
        a: &PyCoreLinkPoint,
        b: &PyCoreLinkPoint,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    ) -> PyResult<PyLinkId> {
        check_link_point(&self.inner, a.inner)?;
        check_link_point(&self.inner, b.inner)?;
        check_nonneg(rest_length, "rest_length")?;
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(damping, "damping")?;
        Ok(PyLinkId {
            inner: self
                .inner
                .add_spring(a.inner, b.inner, rest_length, stiffness, damping),
        })
    }

    fn add_ball_joint(
        &mut self,
        a: &PyCoreLinkPoint,
        b: &PyCoreLinkPoint,
        stiffness: f64,
    ) -> PyResult<PyLinkId> {
        check_link_point(&self.inner, a.inner)?;
        check_link_point(&self.inner, b.inner)?;
        check_nonneg(stiffness, "stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_ball_joint(a.inner, b.inner, stiffness),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), rest_rotation = (0.0, 0.0, 0.0, 1.0), stiffness = 1e6, rotational_stiffness = 5e4))]
    #[allow(clippy::too_many_arguments)]
    fn add_weld(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        rest_rotation: QuatArg,
        stiffness: f64,
        rotational_stiffness: f64,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(rotational_stiffness, "rotational_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_weld(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                to_quat(rest_rotation)?,
                stiffness,
                rotational_stiffness,
            ),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), axis_a = (0.0, 0.0, 1.0), rest_rotation = (0.0, 0.0, 0.0, 1.0), stiffness = 1e6, angular_stiffness = 5e4, limits = None, drive = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_hinge(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        axis_a: Vec3Arg,
        rest_rotation: QuatArg,
        stiffness: f64,
        angular_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: Option<PyCoreJointDrive>,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        let ax = to_vec3(axis_a)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("hinge axis cannot be zero"))?;
        if let Some((lo, hi)) = limits {
            if !lo.is_finite() || !hi.is_finite() || hi < lo {
                return Err(PyValueError::new_err("invalid joint limits"));
            }
        }
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(angular_stiffness, "angular_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_hinge(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                ax,
                to_quat(rest_rotation)?,
                stiffness,
                angular_stiffness,
                limits,
                drive.map_or(JointDrive::NONE, |d| d.inner),
            ),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), axis_b = (1.0, 0.0, 0.0), rest_rotation = (0.0, 0.0, 0.0, 1.0), stiffness = 1e6, rotational_stiffness = 5e4, limits = None, drive = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_slider(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        axis_b: Vec3Arg,
        rest_rotation: QuatArg,
        stiffness: f64,
        rotational_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: Option<PyCoreJointDrive>,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        let ax = to_vec3(axis_b)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("slider axis cannot be zero"))?;
        if let Some((lo, hi)) = limits {
            if !lo.is_finite() || !hi.is_finite() || hi < lo {
                return Err(PyValueError::new_err("invalid slider limits"));
            }
        }
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(rotational_stiffness, "rotational_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_slider(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                ax,
                to_quat(rest_rotation)?,
                stiffness,
                rotational_stiffness,
                limits,
                drive.map_or(JointDrive::NONE, |d| d.inner),
            ),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), axis_a = (0.0, 1.0, 0.0), rest_rotation = (0.0, 0.0, 0.0, 1.0), stiffness = 1e6, angular_stiffness = 5e4, linear_limits = None, angular_limits = None, linear_drive = None, angular_drive = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_cylindrical(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        axis_a: Vec3Arg,
        rest_rotation: QuatArg,
        stiffness: f64,
        angular_stiffness: f64,
        linear_limits: Option<(f64, f64)>,
        angular_limits: Option<(f64, f64)>,
        linear_drive: Option<PyCoreJointDrive>,
        angular_drive: Option<PyCoreJointDrive>,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        let ax = to_vec3(axis_a)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("cylindrical axis cannot be zero"))?;
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(angular_stiffness, "angular_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_cylindrical(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                ax,
                to_quat(rest_rotation)?,
                stiffness,
                angular_stiffness,
                linear_limits,
                angular_limits,
                linear_drive.map_or(JointDrive::NONE, |d| d.inner),
                angular_drive.map_or(JointDrive::NONE, |d| d.inner),
            ),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), axis_a = (1.0, 0.0, 0.0), axis_b = (0.0, 1.0, 0.0), stiffness = 1e6, angular_stiffness = 5e4))]
    #[allow(clippy::too_many_arguments)]
    fn add_universal(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        axis_a: Vec3Arg,
        axis_b: Vec3Arg,
        stiffness: f64,
        angular_stiffness: f64,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        let ax_a = to_vec3(axis_a)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("universal axis_a cannot be zero"))?;
        let ax_b = to_vec3(axis_b)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("universal axis_b cannot be zero"))?;
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(angular_stiffness, "angular_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_universal(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                ax_a,
                ax_b,
                stiffness,
                angular_stiffness,
            ),
        })
    }

    #[pyo3(signature = (a, b = None, local_a = (0.0, 0.0, 0.0), local_b = (0.0, 0.0, 0.0), normal_a = (0.0, 0.0, 1.0), normal_b = (0.0, 0.0, 1.0), stiffness = 1e6, angular_stiffness = 5e4))]
    #[allow(clippy::too_many_arguments)]
    fn add_planar(
        &mut self,
        a: PyObjectId,
        b: Option<PyObjectId>,
        local_a: Vec3Arg,
        local_b: Vec3Arg,
        normal_a: Vec3Arg,
        normal_b: Vec3Arg,
        stiffness: f64,
        angular_stiffness: f64,
    ) -> PyResult<PyLinkId> {
        check_rigid_or_fixed(&self.inner, a.inner)?;
        if let Some(ob) = b {
            check_rigid_or_fixed(&self.inner, ob.inner)?;
        }
        let na = to_vec3(normal_a)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("planar normal_a cannot be zero"))?;
        let nb = to_vec3(normal_b)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("planar normal_b cannot be zero"))?;
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(angular_stiffness, "angular_stiffness")?;
        Ok(PyLinkId {
            inner: self.inner.add_planar(
                a.inner,
                b.map(|id| id.inner),
                to_vec3(local_a)?,
                to_vec3(local_b)?,
                na,
                nb,
                stiffness,
                angular_stiffness,
            ),
        })
    }

    #[pyo3(signature = (waypoints, stiffness = 0.0, damping = 0.0, slack_only = true, drive = None))]
    fn add_tendon(
        &mut self,
        waypoints: Vec<PyRef<'_, PyCoreLinkPoint>>,
        stiffness: f64,
        damping: f64,
        slack_only: bool,
        drive: Option<PyCoreJointDrive>,
    ) -> PyResult<PyLinkId> {
        if waypoints.len() < 2 {
            return Err(PyValueError::new_err(
                "tendon requires at least 2 waypoints",
            ));
        }
        check_nonneg(stiffness, "stiffness")?;
        check_nonneg(damping, "damping")?;
        let pts: Vec<LinkPoint> = waypoints
            .iter()
            .map(|p| check_link_point(&self.inner, p.inner))
            .collect::<PyResult<_>>()?;
        Ok(PyLinkId {
            inner: self.inner.add_tendon(
                pts,
                stiffness,
                damping,
                slack_only,
                drive.map_or(JointDrive::NONE, |d| d.inner),
            ),
        })
    }

    fn add_joint_probe(&mut self, link: PyLinkId) -> PyResult<PyProbeId> {
        check_link_id(&self.inner, link.inner)?;
        Ok(PyProbeId {
            inner: self.inner.add_probe(Probe::JointState { link: link.inner }),
        })
    }

    fn add_wrench_probe(&mut self, link: PyLinkId) -> PyResult<PyProbeId> {
        check_link_id(&self.inner, link.inner)?;
        Ok(PyProbeId {
            inner: self.inner.add_probe(Probe::Wrench { link: link.inner }),
        })
    }

    #[pyo3(signature = (object, local_origin = (0.0, 0.0, 0.0), local_normal = (0.0, 1.0, 0.0), radius = 0.05))]
    fn add_tactile_probe(
        &mut self,
        object: PyObjectId,
        local_origin: Vec3Arg,
        local_normal: Vec3Arg,
        radius: f64,
    ) -> PyResult<PyProbeId> {
        check_object_id(&self.inner, object.inner)?;
        check_pos(radius, "radius")?;
        let n = to_vec3(local_normal)?
            .try_normalize()
            .ok_or_else(|| PyValueError::new_err("tactile probe normal cannot be zero"))?;
        Ok(PyProbeId {
            inner: self.inner.add_probe(Probe::Tactile {
                object: object.inner,
                local_origin: to_vec3(local_origin)?,
                local_normal: n,
                radius,
            }),
        })
    }

    #[pyo3(signature = (object, local = (0.0, 0.0, 0.0)))]
    fn add_imu_probe(&mut self, object: PyObjectId, local: Vec3Arg) -> PyResult<PyProbeId> {
        check_rigid_or_fixed(&self.inner, object.inner)?;
        Ok(PyProbeId {
            inner: self.inner.add_probe(Probe::Imu {
                object: object.inner,
                local: to_vec3(local)?,
            }),
        })
    }

    #[pyo3(signature = (object, linear, angular = (0.0, 0.0, 0.0)))]
    fn set_velocity(
        &mut self,
        object: PyObjectId,
        linear: Vec3Arg,
        angular: Vec3Arg,
    ) -> PyResult<()> {
        check_object_id(&self.inner, object.inner)?;
        if matches!(
            self.inner.object(object.inner).representation,
            Representation::Fixed(_)
        ) {
            return Err(PyValueError::new_err(
                "cannot set velocity on a fixed object",
            ));
        }
        self.inner
            .set_velocity(object.inner, to_vec3(linear)?, to_vec3(angular)?);
        Ok(())
    }

    #[pyo3(signature = (object, pos, quat = (0.0, 0.0, 0.0, 1.0)))]
    fn set_pose(&mut self, object: PyObjectId, pos: Vec3Arg, quat: QuatArg) -> PyResult<()> {
        check_object_id(&self.inner, object.inner)?;
        let pose = to_pose(pos, quat)?;
        match &mut self.inner.object_mut(object.inner).representation {
            Representation::Fixed(f) => f.pose = pose,
            Representation::Rigid(r) => r.pose = pose,
            Representation::Solid(_) => {
                return Err(PyValueError::new_err(
                    "set_pose is only supported on fixed or rigid objects",
                ))
            }
        }
        Ok(())
    }

    fn set_joint_target(&mut self, link: PyLinkId, target: f64) -> PyResult<()> {
        check_link_id(&self.inner, link.inner)?;
        if !target.is_finite() {
            return Err(PyValueError::new_err("target must be finite"));
        }
        let supported = matches!(
            self.inner.link(link.inner),
            Link::Hinge { .. }
                | Link::Slider { .. }
                | Link::Cylindrical { .. }
                | Link::Tendon { .. }
                | Link::Spring { .. }
        );
        if !supported {
            return Err(PyValueError::new_err(
                "set_joint_target requires Hinge, Slider, Cylindrical, Tendon, or Spring",
            ));
        }
        self.inner.set_joint_target(link.inner, target);
        Ok(())
    }

    fn set_cylindrical_targets(
        &mut self,
        link: PyLinkId,
        linear_target: f64,
        angular_target: f64,
    ) -> PyResult<()> {
        check_link_id(&self.inner, link.inner)?;
        if !linear_target.is_finite() || !angular_target.is_finite() {
            return Err(PyValueError::new_err("targets must be finite"));
        }
        if !matches!(self.inner.link(link.inner), Link::Cylindrical { .. }) {
            return Err(PyValueError::new_err(
                "set_cylindrical_targets requires a Cylindrical link",
            ));
        }
        self.inner
            .set_cylindrical_targets(link.inner, linear_target, angular_target);
        Ok(())
    }

    fn set_joint_drive(&mut self, link: PyLinkId, drive: &PyCoreJointDrive) -> PyResult<()> {
        check_link_id(&self.inner, link.inner)?;
        let supported = matches!(
            self.inner.link(link.inner),
            Link::Hinge { .. } | Link::Slider { .. } | Link::Tendon { .. }
        );
        if !supported {
            return Err(PyValueError::new_err(
                "set_joint_drive requires Hinge, Slider, or Tendon",
            ));
        }
        self.inner.set_joint_drive(link.inner, drive.inner);
        Ok(())
    }

    fn joint_coordinate(&self, link: PyLinkId) -> PyResult<f64> {
        check_link_id(&self.inner, link.inner)?;
        self.inner
            .joint_coordinate(link.inner)
            .ok_or_else(|| PyValueError::new_err("link type has no scalar joint_coordinate"))
    }

    fn joint_velocity(&self, link: PyLinkId) -> PyResult<f64> {
        check_link_id(&self.inner, link.inner)?;
        self.inner
            .joint_velocity(link.inner)
            .ok_or_else(|| PyValueError::new_err("link type has no scalar joint_velocity"))
    }

    fn object_position(&self, object: PyObjectId) -> PyResult<Vec3Arg> {
        check_object_id(&self.inner, object.inner)?;
        let p = match &self.inner.object(object.inner).representation {
            Representation::Fixed(f) => f.pose.position,
            Representation::Rigid(r) => r.pose.position,
            Representation::Solid(_) => self
                .inner
                .object(object.inner)
                .centre_of_mass()
                .unwrap_or(Vec3::ZERO),
        };
        Ok(from_vec3(p))
    }

    fn object_orientation(&self, object: PyObjectId) -> PyResult<QuatArg> {
        check_object_id(&self.inner, object.inner)?;
        match &self.inner.object(object.inner).representation {
            Representation::Fixed(f) => Ok(from_quat(f.pose.rotation)),
            Representation::Rigid(r) => Ok(from_quat(r.pose.rotation)),
            Representation::Solid(_) => Err(PyValueError::new_err(
                "deformable solids have per-node positions rather than one orientation",
            )),
        }
    }

    fn object_velocity(&self, object: PyObjectId) -> PyResult<Vec3Arg> {
        check_object_id(&self.inner, object.inner)?;
        match &self.inner.object(object.inner).representation {
            Representation::Fixed(_) => Ok((0.0, 0.0, 0.0)),
            Representation::Rigid(r) => Ok(from_vec3(r.velocity)),
            Representation::Solid(s) => {
                let n = s.nodes.velocity.len().max(1) as f64;
                let mean = s.nodes.velocity.iter().copied().sum::<Vec3>() / n;
                Ok(from_vec3(mean))
            }
        }
    }

    fn object_angular_velocity(&self, object: PyObjectId) -> PyResult<Vec3Arg> {
        check_object_id(&self.inner, object.inner)?;
        match &self.inner.object(object.inner).representation {
            Representation::Fixed(_) => Ok((0.0, 0.0, 0.0)),
            Representation::Rigid(r) => Ok(from_vec3(r.spin)),
            Representation::Solid(_) => Err(PyValueError::new_err(
                "deformable solids have per-node velocities",
            )),
        }
    }

    fn solid_node_positions(&self, object: PyObjectId) -> PyResult<Vec<Vec3Arg>> {
        check_object_id(&self.inner, object.inner)?;
        let Representation::Solid(s) = &self.inner.object(object.inner).representation else {
            return Err(PyValueError::new_err("object is not a deformable solid"));
        };
        Ok(s.nodes.position.iter().copied().map(from_vec3).collect())
    }

    // --- Fusion-VLA additions: what an external renderer needs. ---

    /// Every object's id, in index order.
    fn object_ids(&self) -> Vec<PyObjectId> {
        (0..self.inner.objects().len() as u32)
            .map(|i| PyObjectId { inner: ObjectId(i) })
            .collect()
    }

    /// The object's name (from URDF/MJCF link or geom names, or `add_*`).
    fn object_name(&self, object: PyObjectId) -> PyResult<String> {
        check_object_id(&self.inner, object.inner)?;
        Ok(self.inner.object(object.inner).name.clone())
    }

    /// `"fixed"`, `"rigid"` or `"solid"`.
    fn object_kind(&self, object: PyObjectId) -> PyResult<&'static str> {
        check_object_id(&self.inner, object.inner)?;
        Ok(match &self.inner.object(object.inner).representation {
            Representation::Fixed(_) => "fixed",
            Representation::Rigid(_) => "rigid",
            Representation::Solid(_) => "solid",
        })
    }

    /// The collision shape in the object's own frame (the frame given by
    /// `object_position` / `object_orientation`), as a dict:
    /// `{"type": "sphere", "radius"}`, `{"type": "box", "half_extents"}`,
    /// `{"type": "capsule", "radius", "half_length"}` (axis = local +y), or
    /// `{"type": "halfspace", "normal", "offset"}`. `None` for FEM solids,
    /// which are rendered from `solid_node_positions`.
    fn object_shape<'py>(
        &self,
        py: Python<'py>,
        object: PyObjectId,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        check_object_id(&self.inner, object.inner)?;
        let shape = match &self.inner.object(object.inner).representation {
            Representation::Fixed(f) => f.shape,
            Representation::Rigid(r) => r.shape,
            Representation::Solid(_) => return Ok(None),
        };
        let d = PyDict::new(py);
        match shape {
            Shape::Sphere { radius } => {
                d.set_item("type", "sphere")?;
                d.set_item("radius", radius)?;
            }
            Shape::Cuboid { half } => {
                d.set_item("type", "box")?;
                d.set_item("half_extents", from_vec3(half))?;
            }
            Shape::Capsule {
                radius,
                half_length,
            } => {
                d.set_item("type", "capsule")?;
                d.set_item("radius", radius)?;
                d.set_item("half_length", half_length)?;
            }
            Shape::HalfSpace { normal, offset } => {
                d.set_item("type", "halfspace")?;
                d.set_item("normal", from_vec3(normal))?;
                d.set_item("offset", offset)?;
            }
        }
        Ok(Some(d))
    }
}

#[pyclass(module = "fusion_vla.core", name = "Settings")]
#[derive(Clone)]
pub struct PyCoreSettings {
    pub(crate) inner: Settings,
}

#[pymethods]
impl PyCoreSettings {
    #[new]
    #[pyo3(signature = (
        integrator = "tr_bdf2",
        tolerance = 1e-3,
        max_iterations = 100,
        friction_updates = 1,
        impact_law = true,
        self_contact = true,
        islands = true,
        sleeping = true,
        parallel_islands = true,
        energy_accounting = false,
        contact_precision = None,
        settle_time = Some(0.1),
        solid_block_size = 1,
        max_substep = None,
        cfl_subcycling = None,
        augmented_lagrangian_passes = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        integrator: &str,
        tolerance: f64,
        max_iterations: usize,
        friction_updates: usize,
        impact_law: bool,
        self_contact: bool,
        islands: bool,
        sleeping: bool,
        parallel_islands: bool,
        energy_accounting: bool,
        contact_precision: Option<f64>,
        settle_time: Option<f64>,
        solid_block_size: usize,
        max_substep: Option<f64>,
        cfl_subcycling: Option<f64>,
        augmented_lagrangian_passes: usize,
    ) -> PyResult<Self> {
        if let Some(cp) = contact_precision {
            check_pos(cp, "contact_precision")?;
        }
        if let Some(st) = settle_time {
            check_pos(st, "settle_time")?;
        }
        if !(1..=16).contains(&solid_block_size) {
            return Err(PyValueError::new_err("solid_block_size must be in 1..=16"));
        }
        if let Some(ms) = max_substep {
            check_pos(ms, "max_substep")?;
        }
        if let Some(cfl) = cfl_subcycling {
            check_pos(cfl, "cfl_subcycling")?;
        }
        let s = Settings {
            integrator: parse_integrator(integrator)?,
            tolerance: check_pos(tolerance, "tolerance")?,
            max_iterations: max_iterations.max(1),
            friction_updates,
            impact_law,
            self_contact,
            islands,
            sleeping,
            parallel_islands,
            energy_accounting,
            contact_precision,
            settle_time,
            solid_block_size,
            max_substep,
            cfl_subcycling,
            augmented_lagrangian_passes,
            ..Settings::default()
        };
        Ok(Self { inner: s })
    }

    #[getter]
    fn integrator(&self) -> &'static str {
        integrator_name(self.inner.integrator)
    }
    #[setter]
    fn set_integrator(&mut self, name: &str) -> PyResult<()> {
        self.inner.integrator = parse_integrator(name)?;
        Ok(())
    }

    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }
    #[setter]
    fn set_tolerance(&mut self, val: f64) -> PyResult<()> {
        self.inner.tolerance = check_pos(val, "tolerance")?;
        Ok(())
    }

    #[getter]
    fn augmented_lagrangian_passes(&self) -> usize {
        self.inner.augmented_lagrangian_passes
    }
    #[setter]
    fn set_augmented_lagrangian_passes(&mut self, val: usize) {
        self.inner.augmented_lagrangian_passes = val;
    }

    #[getter]
    fn islands(&self) -> bool {
        self.inner.islands
    }
    #[setter]
    fn set_islands(&mut self, val: bool) {
        self.inner.islands = val;
    }

    #[getter]
    fn sleeping(&self) -> bool {
        self.inner.sleeping
    }
    #[setter]
    fn set_sleeping(&mut self, val: bool) {
        self.inner.sleeping = val;
    }
}

#[pyclass(module = "fusion_vla.core", name = "StepReport")]
#[derive(Clone, Copy)]
pub struct PyCoreStepReport {
    inner: StepReport,
}

#[pymethods]
impl PyCoreStepReport {
    #[getter]
    fn failed(&self) -> bool {
        self.inner.failed()
    }
    #[getter]
    fn unconverged(&self) -> bool {
        self.inner.unconverged
    }
    #[getter]
    fn invalid(&self) -> bool {
        self.inner.invalid
    }
    #[getter]
    fn solves(&self) -> usize {
        self.inner.solves
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
    #[getter]
    fn linear_iterations(&self) -> usize {
        self.inner.linear_iterations
    }
    #[getter]
    fn grips(&self) -> usize {
        self.inner.grips
    }
    #[getter]
    fn impacts(&self) -> usize {
        self.inner.impacts
    }
    #[getter]
    fn splits(&self) -> usize {
        self.inner.splits
    }
    #[getter]
    fn islands(&self) -> usize {
        self.inner.islands
    }
    #[getter]
    fn sleeping_islands(&self) -> usize {
        self.inner.sleeping_islands
    }
    #[getter]
    fn max_island_substeps(&self) -> usize {
        self.inner.max_island_substeps
    }
}

#[pyclass(module = "fusion_vla.core", name = "ProbeReading")]
#[derive(Clone)]
pub struct PyCoreProbeReading {
    inner: ProbeReading,
}

fn reading_as_vec(r: ProbeReading) -> Vec<f64> {
    match r {
        ProbeReading::JointState {
            coordinate,
            velocity,
        } => vec![coordinate, velocity],
        ProbeReading::Wrench { force, torque } => {
            vec![force.x, force.y, force.z, torque.x, torque.y, torque.z]
        }
        ProbeReading::Tactile {
            normal_force,
            total_force,
            center_of_pressure,
            active_contacts,
        } => {
            let cop = center_of_pressure.unwrap_or(Vec3::ZERO);
            vec![
                normal_force,
                total_force.x,
                total_force.y,
                total_force.z,
                cop.x,
                cop.y,
                cop.z,
                active_contacts as f64,
            ]
        }
        ProbeReading::Imu {
            linear_acceleration,
            angular_velocity,
        } => vec![
            linear_acceleration.x,
            linear_acceleration.y,
            linear_acceleration.z,
            angular_velocity.x,
            angular_velocity.y,
            angular_velocity.z,
        ],
    }
}

#[pymethods]
impl PyCoreProbeReading {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ProbeReading::JointState { .. } => "joint_state",
            ProbeReading::Wrench { .. } => "wrench",
            ProbeReading::Tactile { .. } => "tactile",
            ProbeReading::Imu { .. } => "imu",
        }
    }

    #[getter]
    fn coordinate(&self) -> Option<f64> {
        match self.inner {
            ProbeReading::JointState { coordinate, .. } => Some(coordinate),
            _ => None,
        }
    }

    #[getter]
    fn velocity(&self) -> Option<f64> {
        match self.inner {
            ProbeReading::JointState { velocity, .. } => Some(velocity),
            _ => None,
        }
    }

    #[getter]
    fn force(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Wrench { force, .. } => Some(from_vec3(force)),
            _ => None,
        }
    }

    #[getter]
    fn torque(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Wrench { torque, .. } => Some(from_vec3(torque)),
            _ => None,
        }
    }

    #[getter]
    fn normal_force(&self) -> Option<f64> {
        match self.inner {
            ProbeReading::Tactile { normal_force, .. } => Some(normal_force),
            _ => None,
        }
    }

    #[getter]
    fn total_force(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Tactile { total_force, .. } => Some(from_vec3(total_force)),
            _ => None,
        }
    }

    #[getter]
    fn center_of_pressure(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Tactile {
                center_of_pressure, ..
            } => center_of_pressure.map(from_vec3),
            _ => None,
        }
    }

    #[getter]
    fn active_contacts(&self) -> Option<usize> {
        match self.inner {
            ProbeReading::Tactile {
                active_contacts, ..
            } => Some(active_contacts),
            _ => None,
        }
    }

    #[getter]
    fn linear_acceleration(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Imu {
                linear_acceleration,
                ..
            } => Some(from_vec3(linear_acceleration)),
            _ => None,
        }
    }

    #[getter]
    fn angular_velocity(&self) -> Option<Vec3Arg> {
        match self.inner {
            ProbeReading::Imu {
                angular_velocity, ..
            } => Some(from_vec3(angular_velocity)),
            _ => None,
        }
    }

    fn as_vector(&self) -> Vec<f64> {
        reading_as_vec(self.inner)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("kind", self.kind())?;
        match self.inner {
            ProbeReading::JointState {
                coordinate,
                velocity,
            } => {
                d.set_item("coordinate", coordinate)?;
                d.set_item("velocity", velocity)?;
            }
            ProbeReading::Wrench { force, torque } => {
                d.set_item("force", from_vec3(force))?;
                d.set_item("torque", from_vec3(torque))?;
            }
            ProbeReading::Tactile {
                normal_force,
                total_force,
                center_of_pressure,
                active_contacts,
            } => {
                d.set_item("normal_force", normal_force)?;
                d.set_item("total_force", from_vec3(total_force))?;
                d.set_item("center_of_pressure", center_of_pressure.map(from_vec3))?;
                d.set_item("active_contacts", active_contacts)?;
            }
            ProbeReading::Imu {
                linear_acceleration,
                angular_velocity,
            } => {
                d.set_item("linear_acceleration", from_vec3(linear_acceleration))?;
                d.set_item("angular_velocity", from_vec3(angular_velocity))?;
            }
        }
        Ok(d)
    }
}

#[pyclass(module = "fusion_vla.core", name = "AdjointLoss")]
#[derive(Clone, Default)]
pub struct PyCoreAdjointLoss {
    inner: AdjointLoss,
}

#[pymethods]
impl PyCoreAdjointLoss {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn set_d_loss_dx(&mut self, grads: Vec<f64>) {
        self.inner.d_loss_dx = grads;
    }

    fn add_object_position_grad(&mut self, object: PyObjectId, grad: Vec3Arg) -> PyResult<()> {
        self.inner
            .object_position_grads
            .push((object.inner, to_vec3(grad)?));
        Ok(())
    }

    fn add_object_velocity_grad(&mut self, object: PyObjectId, grad: Vec3Arg) -> PyResult<()> {
        self.inner
            .object_velocity_grads
            .push((object.inner, to_vec3(grad)?));
        Ok(())
    }

    fn add_solid_node_position_grad(
        &mut self,
        object: PyObjectId,
        node: usize,
        grad: Vec3Arg,
    ) -> PyResult<()> {
        self.inner
            .solid_node_position_grads
            .push((object.inner, node, to_vec3(grad)?));
        Ok(())
    }

    fn add_solid_node_velocity_grad(
        &mut self,
        object: PyObjectId,
        node: usize,
        grad: Vec3Arg,
    ) -> PyResult<()> {
        self.inner
            .solid_node_velocity_grads
            .push((object.inner, node, to_vec3(grad)?));
        Ok(())
    }
}

#[pyclass(module = "fusion_vla.core", name = "AdjointGradients")]
pub struct PyCoreAdjointGradients {
    inner: AdjointGradients,
}

#[pymethods]
impl PyCoreAdjointGradients {
    #[getter]
    fn world_after(&self) -> PyCoreWorld {
        PyCoreWorld {
            inner: self.inner.world_after.clone(),
        }
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn adjoint(&self) -> Vec<f64> {
        self.inner.adjoint.clone()
    }

    #[getter]
    fn d_initial_positions(&self) -> Vec<f64> {
        self.inner.d_initial_positions.clone()
    }

    #[getter]
    fn d_initial_velocities(&self) -> Vec<f64> {
        self.inner.d_initial_velocities.clone()
    }

    #[getter]
    fn d_gravity(&self) -> Vec3Arg {
        from_vec3(self.inner.d_gravity)
    }

    #[getter]
    fn d_joint_targets(&self) -> Vec<f64> {
        self.inner.d_joint_targets.clone()
    }

    #[getter]
    fn d_joint_stiffnesses(&self) -> Vec<f64> {
        self.inner.d_joint_stiffnesses.clone()
    }

    fn joint_target_grad(&self, link: PyLinkId) -> PyResult<f64> {
        self.inner
            .d_joint_targets
            .get(link.inner.0 as usize)
            .copied()
            .ok_or_else(|| PyIndexError::new_err(format!("LinkId({}) out of range", link.inner.0)))
    }

    fn joint_stiffness_grad(&self, link: PyLinkId) -> PyResult<f64> {
        self.inner
            .d_joint_stiffnesses
            .get(link.inner.0 as usize)
            .copied()
            .ok_or_else(|| PyIndexError::new_err(format!("LinkId({}) out of range", link.inner.0)))
    }
}

#[pyclass(module = "fusion_vla.core", name = "Mechanics")]
pub struct PyCoreMechanics {
    inner: Mechanics,
    pool: Pool,
    prepared: bool,
}

#[pymethods]
impl PyCoreMechanics {
    #[new]
    #[pyo3(signature = (settings = None, threads = None))]
    fn new(settings: Option<PyCoreSettings>, threads: Option<usize>) -> PyResult<Self> {
        let pool = match threads {
            Some(0) => return Err(PyValueError::new_err("threads must be >= 1")),
            Some(n) => Pool::new(n),
            None => Pool::with_default_threads(),
        };
        let cfg = settings.map_or_else(Settings::default, |s| s.inner);
        Ok(Self {
            inner: Mechanics::new(cfg),
            pool,
            prepared: false,
        })
    }

    fn prepare(&mut self, world: &PyCoreWorld) {
        self.inner.prepare(&self.pool, &world.inner);
        self.prepared = true;
    }

    fn invalidate(&mut self) {
        self.inner.invalidate();
        self.prepared = false;
    }

    fn step(
        &mut self,
        py: Python<'_>,
        world: &mut PyCoreWorld,
        dt: f64,
    ) -> PyResult<PyCoreStepReport> {
        check_pos(dt, "dt")?;
        let solver = &mut self.inner;
        let pool = &self.pool;
        let w = &mut world.inner;
        let report = py.allow_threads(|| solver.step(pool, w, dt));
        self.prepared = true;
        Ok(PyCoreStepReport { inner: report })
    }

    fn sample_probes(&mut self, py: Python<'_>, world: &PyCoreWorld) -> Vec<PyCoreProbeReading> {
        let solver = &mut self.inner;
        let pool = &self.pool;
        let w = &world.inner;
        let raw = py.allow_threads(|| solver.sample_probes(pool, w));
        raw.into_iter()
            .map(|inner| PyCoreProbeReading { inner })
            .collect()
    }

    fn backward_step(
        &mut self,
        py: Python<'_>,
        world_before: &PyCoreWorld,
        dt: f64,
        loss: &PyCoreAdjointLoss,
    ) -> PyResult<PyCoreAdjointGradients> {
        check_pos(dt, "dt")?;
        for &(id, _) in &loss.inner.object_position_grads {
            check_object_id(&world_before.inner, id)?;
        }
        for &(id, _) in &loss.inner.object_velocity_grads {
            check_object_id(&world_before.inner, id)?;
        }
        for &(id, _, _) in &loss.inner.solid_node_position_grads {
            check_object_id(&world_before.inner, id)?;
        }
        for &(id, _, _) in &loss.inner.solid_node_velocity_grads {
            check_object_id(&world_before.inner, id)?;
        }
        let solver = &mut self.inner;
        let pool = &self.pool;
        let wb = &world_before.inner;
        let loss_ref = &loss.inner;
        let grads = py.allow_threads(|| solver.backward_step(pool, wb, dt, loss_ref));
        self.prepared = true;
        Ok(PyCoreAdjointGradients { inner: grads })
    }

    fn strain_energy(&mut self, world: &PyCoreWorld) -> f64 {
        self.inner.strain_energy(&self.pool, &world.inner)
    }

    fn checkpoint_bytes(&mut self, world: &PyCoreWorld) -> Vec<u8> {
        if !self.prepared {
            self.inner.prepare(&self.pool, &world.inner);
            self.prepared = true;
        }
        self.inner.checkpoint(&world.inner).to_bytes()
    }

    fn restore_checkpoint_bytes(&mut self, world: &mut PyCoreWorld, bytes: &[u8]) -> PyResult<()> {
        if !self.prepared {
            self.inner.prepare(&self.pool, &world.inner);
            self.prepared = true;
        }
        let template = self.inner.checkpoint(&world.inner);
        let state = MechanicsState::from_bytes(bytes, &template).map_err(PyValueError::new_err)?;
        self.inner.restore_checkpoint(&mut world.inner, &state);
        self.prepared = true;
        Ok(())
    }
}

#[pyclass(module = "fusion_vla.core", name = "DomainRandomizer")]
#[derive(Clone)]
pub struct PyCoreDomainRandomizer {
    inner: DomainRandomizer,
}

fn check_range(range: (f64, f64), name: &str, positive_min: bool) -> PyResult<(f64, f64)> {
    if !range.0.is_finite() || !range.1.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} bounds must be finite"
        )));
    }
    let min_val = range.0.min(range.1);
    if positive_min {
        if min_val <= 0.0 {
            return Err(PyValueError::new_err(format!("{name} bounds must be > 0")));
        }
    } else if min_val < 0.0 {
        return Err(PyValueError::new_err(format!("{name} bounds must be >= 0")));
    }
    Ok(range)
}

#[pymethods]
impl PyCoreDomainRandomizer {
    #[new]
    #[pyo3(signature = (
        seed = 1,
        mass_scale = (1.0, 1.0),
        friction_scale = (1.0, 1.0),
        stiffness_scale = (1.0, 1.0),
        damping_scale = (1.0, 1.0),
        gravity_jitter = (0.0, 0.0, 0.0),
    ))]
    fn new(
        seed: u64,
        mass_scale: (f64, f64),
        friction_scale: (f64, f64),
        stiffness_scale: (f64, f64),
        damping_scale: (f64, f64),
        gravity_jitter: Vec3Arg,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DomainRandomizer {
                seed,
                mass_scale: check_range(mass_scale, "mass_scale", true)?,
                friction_scale: check_range(friction_scale, "friction_scale", false)?,
                stiffness_scale: check_range(stiffness_scale, "stiffness_scale", false)?,
                damping_scale: check_range(damping_scale, "damping_scale", false)?,
                gravity_jitter: to_vec3(gravity_jitter)?,
            },
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[setter]
    fn set_seed(&mut self, seed: u64) {
        self.inner.seed = seed;
    }
}

#[pyclass(module = "fusion_vla.core", name = "WorldBatch")]
pub struct PyCoreWorldBatch {
    pub(crate) inner: WorldBatch,
    pool: Pool,
}

impl PyCoreWorldBatch {
    fn check_env(&self, index: usize) -> PyResult<()> {
        if index >= self.inner.len() {
            return Err(PyIndexError::new_err(format!(
                "env_index {index} out of range (batch size {})",
                self.inner.len()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyCoreWorldBatch {
    #[new]
    #[pyo3(signature = (template, count, settings = None, threads = None))]
    fn new(
        template: &PyCoreWorld,
        count: usize,
        settings: Option<PyCoreSettings>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        if count == 0 {
            return Err(PyValueError::new_err("batch count must be >= 1"));
        }
        let pool = match threads {
            Some(0) => return Err(PyValueError::new_err("threads must be >= 1")),
            Some(n) => Pool::new(n),
            None => Pool::with_default_threads(),
        };
        let cfg = settings.map_or_else(Settings::default, |s| s.inner);
        Ok(Self {
            inner: WorldBatch::from_template(template.inner.clone(), cfg, count),
            pool,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn reset_env(&mut self, env_index: usize) -> PyResult<()> {
        self.check_env(env_index)?;
        self.inner.reset_env(env_index);
        Ok(())
    }

    fn reset_all(&mut self) {
        self.inner.reset_all();
    }

    fn reset_envs(&mut self, indices: Vec<usize>) -> PyResult<()> {
        for &idx in &indices {
            self.check_env(idx)?;
        }
        self.inner.reset_envs(&indices);
        Ok(())
    }

    fn randomize_env(&mut self, env_index: usize, config: &PyCoreDomainRandomizer) -> PyResult<()> {
        self.check_env(env_index)?;
        self.inner.randomize_env(env_index, &config.inner);
        Ok(())
    }

    fn randomize_all(&mut self, config: &PyCoreDomainRandomizer) {
        self.inner.randomize_all(&config.inner);
    }

    #[pyo3(signature = (indices, config, seed_offset = 0))]
    fn randomize_envs(
        &mut self,
        indices: Vec<usize>,
        config: &PyCoreDomainRandomizer,
        seed_offset: u64,
    ) -> PyResult<()> {
        for &idx in &indices {
            self.check_env(idx)?;
        }
        self.inner
            .randomize_envs(&indices, &config.inner, seed_offset);
        Ok(())
    }

    fn set_joint_targets(
        &mut self,
        env_index: usize,
        targets: Vec<(PyLinkId, f64)>,
    ) -> PyResult<()> {
        self.check_env(env_index)?;
        let w = self.inner.world(env_index);
        let pairs: Vec<(LinkId, f64)> = targets
            .into_iter()
            .map(|(lid, v)| {
                check_link_id(w, lid.inner)?;
                if !v.is_finite() {
                    return Err(PyValueError::new_err("target must be finite"));
                }
                Ok((lid.inner, v))
            })
            .collect::<PyResult<_>>()?;
        self.inner.set_joint_targets(env_index, &pairs);
        Ok(())
    }

    fn set_batch_joint_targets(
        &mut self,
        links: Vec<PyLinkId>,
        targets_row_major: Vec<f64>,
    ) -> PyResult<()> {
        if let Some(w) = (!self.inner.is_empty()).then(|| self.inner.world(0)) {
            for &lid in &links {
                check_link_id(w, lid.inner)?;
            }
        }
        let expected = self.inner.len() * links.len();
        if targets_row_major.len() != expected {
            return Err(PyValueError::new_err(format!(
                "expected {expected} targets (count * links), got {}",
                targets_row_major.len()
            )));
        }
        if targets_row_major.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("all joint targets must be finite"));
        }
        let raw_links: Vec<LinkId> = links.into_iter().map(|l| l.inner).collect();
        self.inner
            .set_batch_joint_targets(&raw_links, &targets_row_major);
        Ok(())
    }

    fn step_all(&mut self, py: Python<'_>, dt: f64) -> PyResult<()> {
        check_pos(dt, "dt")?;
        let batch = &mut self.inner;
        let pool = &self.pool;
        py.allow_threads(|| batch.step_all(pool, dt));
        Ok(())
    }

    fn sample_all_probes(&mut self, py: Python<'_>) -> Vec<Vec<PyCoreProbeReading>> {
        let batch = &mut self.inner;
        let pool = &self.pool;
        let all = py.allow_threads(|| batch.sample_all_probes(pool));
        all.into_iter()
            .map(|env| {
                env.into_iter()
                    .map(|inner| PyCoreProbeReading { inner })
                    .collect()
            })
            .collect()
    }

    fn sample_observations(&mut self, py: Python<'_>) -> Vec<Vec<f64>> {
        let batch = &mut self.inner;
        let pool = &self.pool;
        let all = py.allow_threads(|| batch.sample_all_probes(pool));
        all.into_iter()
            .map(|env| env.into_iter().flat_map(reading_as_vec).collect())
            .collect()
    }

    fn joint_coordinates(&self, links: Vec<PyLinkId>) -> PyResult<Vec<f64>> {
        if let Some(w) = (!self.inner.is_empty()).then(|| self.inner.world(0)) {
            for &lid in &links {
                check_link_id(w, lid.inner)?;
            }
        }
        let raw_links: Vec<LinkId> = links.into_iter().map(|l| l.inner).collect();
        Ok(self.inner.joint_coordinates(&raw_links))
    }

    fn joint_velocities(&self, links: Vec<PyLinkId>) -> PyResult<Vec<f64>> {
        if let Some(w) = (!self.inner.is_empty()).then(|| self.inner.world(0)) {
            for &lid in &links {
                check_link_id(w, lid.inner)?;
            }
        }
        let raw_links: Vec<LinkId> = links.into_iter().map(|l| l.inner).collect();
        Ok(self.inner.joint_velocities(&raw_links))
    }

    fn object_positions(&self, objects: Vec<PyObjectId>) -> PyResult<Vec<f64>> {
        if let Some(w) = (!self.inner.is_empty()).then(|| self.inner.world(0)) {
            for &oid in &objects {
                check_object_id(w, oid.inner)?;
            }
        }
        let mut out = Vec::with_capacity(self.inner.len() * objects.len() * 3);
        for i in 0..self.inner.len() {
            let w = self.inner.world(i);
            for &oid in &objects {
                let p = match &w.object(oid.inner).representation {
                    Representation::Fixed(f) => f.pose.position,
                    Representation::Rigid(r) => r.pose.position,
                    Representation::Solid(_) => {
                        w.object(oid.inner).centre_of_mass().unwrap_or(Vec3::ZERO)
                    }
                };
                out.extend_from_slice(&[p.x, p.y, p.z]);
            }
        }
        Ok(out)
    }

    /// Fusion-VLA addition: `[px, py, pz, qx, qy, qz, qw]` per object per env,
    /// row-major `(envs, objects, 7)`. Solids report their centre of mass and
    /// an identity rotation.
    fn object_poses(&self, objects: Vec<PyObjectId>) -> PyResult<Vec<f64>> {
        if let Some(w) = (!self.inner.is_empty()).then(|| self.inner.world(0)) {
            for &oid in &objects {
                check_object_id(w, oid.inner)?;
            }
        }
        let mut out = Vec::with_capacity(self.inner.len() * objects.len() * 7);
        for i in 0..self.inner.len() {
            let w = self.inner.world(i);
            for &oid in &objects {
                let (p, q) = match &w.object(oid.inner).representation {
                    Representation::Fixed(f) => (f.pose.position, f.pose.rotation),
                    Representation::Rigid(r) => (r.pose.position, r.pose.rotation),
                    Representation::Solid(_) => (
                        w.object(oid.inner).centre_of_mass().unwrap_or(Vec3::ZERO),
                        Quat::IDENTITY,
                    ),
                };
                out.extend_from_slice(&[p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
            }
        }
        Ok(out)
    }

    fn world_snapshot(&self, env_index: usize) -> PyResult<PyCoreWorld> {
        self.check_env(env_index)?;
        Ok(PyCoreWorld {
            inner: self.inner.world(env_index).clone(),
        })
    }

    #[getter]
    fn reports(&self) -> Vec<PyCoreStepReport> {
        self.inner
            .reports()
            .iter()
            .copied()
            .map(|inner| PyCoreStepReport { inner })
            .collect()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMaterialId>()?;
    module.add_class::<PyObjectId>()?;
    module.add_class::<PyLinkId>()?;
    module.add_class::<PyProbeId>()?;
    module.add_class::<PyCoreLinkPoint>()?;
    module.add_class::<PyCoreMaterial>()?;
    module.add_class::<PyCoreJointDrive>()?;
    module.add_class::<PyCoreRobotModel>()?;
    module.add_class::<PyCoreWorld>()?;
    module.add_class::<PyCoreSettings>()?;
    module.add_class::<PyCoreStepReport>()?;
    module.add_class::<PyCoreProbeReading>()?;
    module.add_class::<PyCoreAdjointLoss>()?;
    module.add_class::<PyCoreAdjointGradients>()?;
    module.add_class::<PyCoreMechanics>()?;
    module.add_class::<PyCoreDomainRandomizer>()?;
    module.add_class::<PyCoreWorldBatch>()?;
    Ok(())
}
