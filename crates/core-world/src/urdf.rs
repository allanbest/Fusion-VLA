//! Pure-Rust (`std`-only) URDF robot parser building articulated bodies and
//! joints directly inside [`World`].

use std::collections::{HashMap, VecDeque};
use std::f64::consts::FRAC_PI_2;

use core_math::{Mat3, Pose, Quat, Vec3};
use core_space::Shape;

use crate::xml::{compose_pose, inverse_pose, parse_vec3, parse_xml_doc, XmlNode};
use crate::{JointDrive, LinkId, LinkPoint, MaterialId, ObjectId, ProbeId, Representation, World};

/// Handles to the spawned [`ObjectId`]s, [`LinkId`]s, [`LinkPoint`]s, and
/// [`ProbeId`]s of an imported URDF or MJCF model.
#[derive(Clone, Debug)]
pub struct RobotModel {
    pub name: String,
    pub root: ObjectId,
    pub links: HashMap<String, ObjectId>,
    pub joints: HashMap<String, LinkId>,
    pub actuators: HashMap<String, LinkId>,
    pub tendons: HashMap<String, LinkId>,
    pub sites: HashMap<String, LinkPoint>,
    pub probes: HashMap<String, ProbeId>,
    pub geoms: HashMap<String, ObjectId>,
}

impl RobotModel {
    pub fn link(&self, name: &str) -> Option<ObjectId> {
        self.links.get(name).copied()
    }

    pub fn joint(&self, name: &str) -> Option<LinkId> {
        self.joints.get(name).copied()
    }

    pub fn actuator(&self, name: &str) -> Option<LinkId> {
        self.actuators.get(name).copied()
    }

    pub fn tendon(&self, name: &str) -> Option<LinkId> {
        self.tendons.get(name).copied()
    }

    pub fn site(&self, name: &str) -> Option<LinkPoint> {
        self.sites.get(name).copied()
    }

    pub fn probe(&self, name: &str) -> Option<ProbeId> {
        self.probes.get(name).copied()
    }

    pub fn geom(&self, name: &str) -> Option<ObjectId> {
        self.geoms.get(name).copied()
    }

    pub fn joint_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.joints.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn actuator_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.actuators.keys().cloned().collect();
        names.sort();
        names
    }
}

fn parse_origin(node: Option<&XmlNode>) -> Result<Pose, String> {
    let Some(n) = node else {
        return Ok(Pose::IDENTITY);
    };
    let xyz = match n.attrs.get("xyz") {
        Some(s) => parse_vec3(s)?,
        None => Vec3::ZERO,
    };
    let rpy = match n.attrs.get("rpy") {
        Some(s) => parse_vec3(s)?,
        None => Vec3::ZERO,
    };
    // URDF fixed-axis XYZ (intrinsic ZYX): R = Rz(yaw) * Ry(pitch) * Rx(roll).
    let q = (Quat::from_axis_angle(Vec3::Z, rpy.z)
        * Quat::from_axis_angle(Vec3::Y, rpy.y)
        * Quat::from_axis_angle(Vec3::X, rpy.x))
    .normalize();
    Ok(Pose::new(xyz, q))
}

#[derive(Clone, Debug)]
struct UrdfLink {
    name: String,
    mass: Option<f64>,
    inertia: Option<Mat3>,
    shape: Shape,
    shape_origin: Pose,
}

#[derive(Clone, Debug)]
struct UrdfJoint {
    name: String,
    kind: String,
    parent: String,
    child: String,
    origin: Pose,
    axis: Vec3,
    limits: Option<(f64, f64)>,
    effort: Option<f64>,
    damping: f64,
}

fn parse_shape(link_node: &XmlNode) -> Result<(Shape, Pose), String> {
    let geom_holder = link_node
        .children
        .iter()
        .find(|c| c.tag == "collision")
        .or_else(|| link_node.children.iter().find(|c| c.tag == "visual"));
    let Some(holder) = geom_holder else {
        return Ok((Shape::Sphere { radius: 0.05 }, Pose::IDENTITY));
    };
    let origin = parse_origin(holder.children.iter().find(|c| c.tag == "origin"))?;
    let Some(geom) = holder.children.iter().find(|c| c.tag == "geometry") else {
        return Ok((Shape::Sphere { radius: 0.05 }, origin));
    };
    if let Some(box_n) = geom.children.iter().find(|c| c.tag == "box") {
        let size = parse_vec3(
            box_n
                .attrs
                .get("size")
                .ok_or("URDF <box> missing size attribute")?,
        )?;
        Ok((Shape::cuboid(size), origin))
    } else if let Some(sph_n) = geom.children.iter().find(|c| c.tag == "sphere") {
        let r = sph_n
            .attrs
            .get("radius")
            .ok_or("URDF <sphere> missing radius")?
            .parse::<f64>()
            .map_err(|_| "invalid sphere radius")?;
        Ok((Shape::Sphere { radius: r }, origin))
    } else if let Some(cyl_n) = geom.children.iter().find(|c| c.tag == "cylinder") {
        let r = cyl_n
            .attrs
            .get("radius")
            .ok_or("URDF <cylinder> missing radius")?
            .parse::<f64>()
            .map_err(|_| "invalid cylinder radius")?;
        let len = cyl_n
            .attrs
            .get("length")
            .ok_or("URDF <cylinder> missing length")?
            .parse::<f64>()
            .map_err(|_| "invalid cylinder length")?;
        // URDF cylinder axis is Z; Fusion Shape::Capsule axis is Y.
        let capsule_rot = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
        let cyl_pose = Pose::new(origin.position, (origin.rotation * capsule_rot).normalize());
        let half_length = (0.5 * len - r).max(0.0);
        Ok((
            Shape::Capsule {
                radius: r,
                half_length,
            },
            cyl_pose,
        ))
    } else {
        Ok((Shape::Sphere { radius: 0.05 }, origin))
    }
}

fn parse_link(node: &XmlNode) -> Result<UrdfLink, String> {
    let name = node
        .attrs
        .get("name")
        .cloned()
        .ok_or("URDF <link> missing name")?;
    let (shape, shape_origin) = parse_shape(node)?;
    let mut mass = None;
    let mut inertia = None;
    if let Some(inertial) = node.children.iter().find(|c| c.tag == "inertial") {
        if let Some(m_node) = inertial.children.iter().find(|c| c.tag == "mass") {
            if let Some(val) = m_node.attrs.get("value") {
                mass = Some(
                    val.parse::<f64>()
                        .map_err(|_| format!("invalid mass '{val}' in link {name}"))?,
                );
            }
        }
        if let Some(i_node) = inertial.children.iter().find(|c| c.tag == "inertia") {
            let f = |k: &str| -> f64 {
                i_node
                    .attrs
                    .get(k)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
            };
            let (ixx, ixy, ixz) = (f("ixx"), f("ixy"), f("ixz"));
            let (iyy, iyz, izz) = (f("iyy"), f("iyz"), f("izz"));
            let mat = Mat3::from_rows(
                Vec3::new(ixx, ixy, ixz),
                Vec3::new(ixy, iyy, iyz),
                Vec3::new(ixz, iyz, izz),
            );
            if ixx > 0.0 && iyy > 0.0 && izz > 0.0 {
                inertia = Some(mat);
            }
        }
    }
    Ok(UrdfLink {
        name,
        mass,
        inertia,
        shape,
        shape_origin,
    })
}

fn parse_joint(node: &XmlNode) -> Result<UrdfJoint, String> {
    let name = node
        .attrs
        .get("name")
        .cloned()
        .ok_or("URDF <joint> missing name")?;
    let kind = node
        .attrs
        .get("type")
        .cloned()
        .ok_or_else(|| format!("URDF <joint name='{name}'> missing type"))?;
    let parent = node
        .children
        .iter()
        .find(|c| c.tag == "parent")
        .and_then(|c| c.attrs.get("link"))
        .cloned()
        .ok_or_else(|| format!("joint {name} missing parent link"))?;
    let child = node
        .children
        .iter()
        .find(|c| c.tag == "child")
        .and_then(|c| c.attrs.get("link"))
        .cloned()
        .ok_or_else(|| format!("joint {name} missing child link"))?;
    let origin = parse_origin(node.children.iter().find(|c| c.tag == "origin"))?;
    let axis = match node
        .children
        .iter()
        .find(|c| c.tag == "axis")
        .and_then(|c| c.attrs.get("xyz"))
    {
        Some(s) => parse_vec3(s)?.try_normalize().unwrap_or(Vec3::Z),
        None => Vec3::X,
    };
    let mut limits = None;
    let mut effort = None;
    if let Some(lim) = node.children.iter().find(|c| c.tag == "limit") {
        let lower = lim.attrs.get("lower").and_then(|s| s.parse::<f64>().ok());
        let upper = lim.attrs.get("upper").and_then(|s| s.parse::<f64>().ok());
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if hi >= lo {
                limits = Some((lo, hi));
            }
        }
        effort = lim.attrs.get("effort").and_then(|s| s.parse::<f64>().ok());
    }
    let damping = node
        .children
        .iter()
        .find(|c| c.tag == "dynamics")
        .and_then(|c| c.attrs.get("damping"))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    Ok(UrdfJoint {
        name,
        kind,
        parent,
        child,
        origin,
        axis,
        limits,
        effort,
        damping,
    })
}

impl World {
    /// Parses a URDF XML robot description and spawns all links and joints into
    /// `self` with base frame at `root_pose`. When `fixed_base` is `true`, the
    /// root link is spawned as a static [`Representation::Fixed`] object.
    pub fn load_urdf(
        &mut self,
        xml: &str,
        material: MaterialId,
        root_pose: Pose,
        fixed_base: bool,
    ) -> Result<RobotModel, String> {
        let robot_node = parse_xml_doc(xml, "robot")?;
        let robot_name = robot_node
            .attrs
            .get("name")
            .cloned()
            .unwrap_or_else(|| "urdf_robot".into());

        let mut urdf_links = HashMap::new();
        let mut urdf_joints = Vec::new();
        for child in &robot_node.children {
            match child.tag.as_str() {
                "link" => {
                    let link = parse_link(child)?;
                    urdf_links.insert(link.name.clone(), link);
                }
                "joint" => {
                    urdf_joints.push(parse_joint(child)?);
                }
                _ => {}
            }
        }
        if urdf_links.is_empty() {
            return Err("URDF has no <link> definitions".into());
        }

        let child_names: HashMap<&str, ()> =
            urdf_joints.iter().map(|j| (j.child.as_str(), ())).collect();
        let root_name = urdf_links
            .keys()
            .find(|k| !child_names.contains_key(k.as_str()))
            .cloned()
            .or_else(|| urdf_links.keys().min().cloned())
            .ok_or("cannot determine URDF root link")?;

        let mut link_world_pose: HashMap<String, Pose> = HashMap::new();
        let mut body_world_pose: HashMap<String, Pose> = HashMap::new();
        let mut spawned_links: HashMap<String, ObjectId> = HashMap::new();
        let mut by_parent: HashMap<&str, Vec<&UrdfJoint>> = HashMap::new();
        for j in &urdf_joints {
            by_parent.entry(j.parent.as_str()).or_default().push(j);
        }

        let mut queue = VecDeque::new();
        link_world_pose.insert(root_name.clone(), root_pose);
        queue.push_back(root_name.clone());

        while let Some(pname) = queue.pop_front() {
            let l_pose = link_world_pose[&pname];
            let l_spec = urdf_links
                .get(&pname)
                .ok_or_else(|| format!("missing link definition '{pname}'"))?;
            let b_pose = compose_pose(l_pose, l_spec.shape_origin);
            body_world_pose.insert(pname.clone(), b_pose);

            let is_root = pname == root_name;
            let obj_id = if is_root && (fixed_base || l_spec.mass == Some(0.0)) {
                self.add_fixed(&pname, material, l_spec.shape, b_pose)
            } else {
                let id = self.add_rigid(&pname, material, l_spec.shape, b_pose);
                if let Representation::Rigid(r) = &mut self.object_mut(id).representation {
                    if let Some(m) = l_spec.mass.filter(|&m| m > 0.0) {
                        r.mass = m;
                    }
                    if let Some(iner) = l_spec.inertia {
                        let rot = l_spec.shape_origin.rotation.to_mat3();
                        r.inertia = rot.transpose() * iner * rot;
                    }
                }
                id
            };
            spawned_links.insert(pname.clone(), obj_id);

            if let Some(children) = by_parent.get(pname.as_str()) {
                for j in children {
                    if !link_world_pose.contains_key(&j.child) {
                        let c_pose = compose_pose(l_pose, j.origin);
                        link_world_pose.insert(j.child.clone(), c_pose);
                        queue.push_back(j.child.clone());
                    }
                }
            }
        }

        let mut spawned_joints: HashMap<String, LinkId> = HashMap::new();
        for j in &urdf_joints {
            let (Some(&p_id), Some(&c_id)) =
                (spawned_links.get(&j.parent), spawned_links.get(&j.child))
            else {
                continue;
            };
            let p_link_pose = link_world_pose[&j.parent];
            let p_body = body_world_pose[&j.parent];
            let c_body = body_world_pose[&j.child];
            let j_world = compose_pose(p_link_pose, j.origin);
            let p_inv = inverse_pose(p_body);
            let c_inv = inverse_pose(c_body);
            let local_a = c_inv.transform_point(j_world.position);
            let local_b = p_inv.transform_point(j_world.position);
            let rel_rot = (p_body.rotation.conjugate() * c_body.rotation).normalize();

            let m_scale = self.object(c_id).mass().clamp(0.1, 100.0);
            let k_lin = 1e6 * m_scale;
            let k_ang = 5e4 * m_scale;

            let axis_world = j_world.rotation.rotate(j.axis).normalize();
            let axis_a = c_inv.rotation.rotate(axis_world).normalize();
            let axis_b = p_inv.rotation.rotate(axis_world).normalize();

            let mut drive = JointDrive::pd(0.0, 0.0, j.damping);
            if let Some(eff) = j.effort {
                if eff > 0.0 && eff.is_finite() {
                    drive = drive.with_max_force(eff);
                }
            }

            let link_id = match j.kind.as_str() {
                "fixed" => self.add_weld(c_id, Some(p_id), local_a, local_b, rel_rot, k_lin, k_ang),
                "revolute" => self.add_hinge(
                    c_id,
                    Some(p_id),
                    local_a,
                    local_b,
                    axis_a,
                    rel_rot,
                    k_lin,
                    k_ang,
                    j.limits,
                    drive,
                ),
                "continuous" => self.add_hinge(
                    c_id,
                    Some(p_id),
                    local_a,
                    local_b,
                    axis_a,
                    rel_rot,
                    k_lin,
                    k_ang,
                    None,
                    drive,
                ),
                "prismatic" => self.add_slider(
                    c_id,
                    Some(p_id),
                    local_a,
                    local_b,
                    axis_b,
                    rel_rot,
                    k_lin,
                    k_ang,
                    j.limits,
                    drive,
                ),
                "planar" => self.add_planar(
                    c_id,
                    Some(p_id),
                    local_a,
                    local_b,
                    axis_a,
                    axis_b,
                    k_lin,
                    k_ang,
                ),
                "floating" => continue,
                other => return Err(format!("unsupported URDF joint type '{other}'")),
            };
            spawned_joints.insert(j.name.clone(), link_id);
        }

        let root = spawned_links[&root_name];
        Ok(RobotModel {
            name: robot_name,
            root,
            links: spawned_links,
            joints: spawned_joints,
            actuators: HashMap::new(),
            tendons: HashMap::new(),
            sites: HashMap::new(),
            probes: HashMap::new(),
            geoms: HashMap::new(),
        })
    }
}
