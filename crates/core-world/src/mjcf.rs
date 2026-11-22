//! Zero-dependency (`std`-only) MuJoCo XML (MJCF) importer building articulated
//! bodies, joints, tendons, contact exclusions, actuators, and probes directly
//! inside [`World`].

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI};

use core_math::{Mat3, Pose, Quat, Vec3};
use core_space::Shape;

use crate::xml::{compose_pose, inverse_pose, parse_f64_list, parse_vec3, parse_xml_doc, XmlNode};
use crate::{
    JointDrive, LinkId, LinkPoint, MaterialId, ObjectId, Probe, ProbeId, Representation,
    RobotModel, World,
};

type AttrMap = HashMap<String, String>;
type ClassTable = HashMap<String, HashMap<String, AttrMap>>;

#[derive(Clone, Debug)]
struct Compiler {
    angle_scale: f64,
    eulerseq: String,
    autolimits: bool,
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            angle_scale: PI / 180.0,
            eulerseq: "xyz".to_string(),
            autolimits: true,
        }
    }
}

#[derive(Clone, Debug)]
struct GeomSpec {
    name: Option<String>,
    shape: Shape,
    local_pose: Pose,
    material: MaterialId,
    mass_override: Option<f64>,
    is_collision: bool,
}

#[derive(Clone, Debug)]
struct JointSpec {
    name: Option<String>,
    kind: String,
    pos: Vec3,
    axis: Vec3,
    limits: Option<(f64, f64)>,
    stiffness: f64,
    damping: f64,
    springref: f64,
}

#[derive(Clone, Debug)]
struct InertialSpec {
    local_pose: Pose,
    mass: Option<f64>,
    inertia: Option<Mat3>,
}

struct MjcfBuilder<'a> {
    world: &'a mut World,
    default_material: MaterialId,
    compiler: Compiler,
    defaults: ClassTable,
    links_by_name: HashMap<String, ObjectId>,
    body_objects: HashMap<String, Vec<ObjectId>>,
    joints_by_name: HashMap<String, LinkId>,
    joint_body: HashMap<String, ObjectId>,
    body_parent_joint: HashMap<ObjectId, LinkId>,
    joint_specs: HashMap<String, JointSpec>,
    actuators_by_name: HashMap<String, LinkId>,
    tendons_by_name: HashMap<String, LinkId>,
    sites: HashMap<String, LinkPoint>,
    site_normals: HashMap<String, (Vec3, f64)>,
    probes_by_name: HashMap<String, ProbeId>,
    geoms_by_name: HashMap<String, ObjectId>,
    material_cache: HashMap<(u64, u64), MaterialId>,
    root_object: Option<ObjectId>,
    unnamed_counter: usize,
}

fn parse_compiler(root: &XmlNode) -> Result<Compiler, String> {
    let mut cfg = Compiler::default();
    for child in &root.children {
        if child.tag != "compiler" {
            continue;
        }
        if let Some(angle) = child.attrs.get("angle") {
            cfg.angle_scale = match angle.as_str() {
                "degree" => PI / 180.0,
                "radian" => 1.0,
                other => return Err(format!("unsupported MJCF <compiler angle='{other}'>")),
            };
        }
        if let Some(seq) = child.attrs.get("eulerseq") {
            let valid = seq.len() == 3
                && (seq.chars().all(|c| matches!(c, 'x' | 'y' | 'z'))
                    || seq.chars().all(|c| matches!(c, 'X' | 'Y' | 'Z')));
            if !valid {
                return Err(format!("invalid MJCF eulerseq '{seq}'"));
            }
            cfg.eulerseq = seq.clone();
        }
        if let Some(auto) = child.attrs.get("autolimits") {
            cfg.autolimits = match auto.as_str() {
                "true" => true,
                "false" => false,
                other => return Err(format!("invalid MJCF autolimits '{other}'")),
            };
        }
    }
    Ok(cfg)
}

fn parse_defaults(node: &XmlNode, parent: &HashMap<String, AttrMap>, table: &mut ClassTable) {
    let mut merged = parent.clone();
    for child in &node.children {
        if child.tag != "default" {
            let entry = merged.entry(child.tag.clone()).or_default();
            for (k, v) in &child.attrs {
                entry.insert(k.clone(), v.clone());
            }
        }
    }
    let class_name = node
        .attrs
        .get("class")
        .cloned()
        .unwrap_or_else(|| "main".to_string());
    table.insert(class_name, merged.clone());
    for child in &node.children {
        if child.tag == "default" {
            parse_defaults(child, &merged, table);
        }
    }
}

fn collect_defaults(root: &XmlNode) -> ClassTable {
    let mut table = ClassTable::new();
    let empty = HashMap::new();
    for child in &root.children {
        if child.tag == "default" {
            parse_defaults(child, &empty, &mut table);
        }
    }
    table
}

fn resolved_attrs(defaults: &ClassTable, node: &XmlNode, active_class: Option<&str>) -> AttrMap {
    let mut out = AttrMap::new();
    if let Some(main) = defaults.get("main").and_then(|m| m.get(&node.tag)) {
        out.extend(main.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    let class_name = node
        .attrs
        .get("class")
        .map(String::as_str)
        .or(active_class)
        .filter(|c| *c != "main");
    if let Some(cls) = class_name {
        if let Some(cls_map) = defaults.get(cls).and_then(|m| m.get(&node.tag)) {
            out.extend(cls_map.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
    }
    for (k, v) in &node.attrs {
        out.insert(k.clone(), v.clone());
    }
    out
}

fn quat_from_axis_char(axis: char, angle: f64) -> Quat {
    match axis.to_ascii_lowercase() {
        'x' => Quat::from_axis_angle(Vec3::X, angle),
        'y' => Quat::from_axis_angle(Vec3::Y, angle),
        'z' => Quat::from_axis_angle(Vec3::Z, angle),
        _ => Quat::IDENTITY,
    }
}

fn quat_from_two_unit_vecs(from: Vec3, to: Vec3) -> Quat {
    let f = from.normalize();
    let t = to.normalize();
    let dot = f.dot(t);
    if dot > 1.0 - 1e-12 {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + 1e-12 {
        let ortho = if f.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let axis = f.cross(ortho).normalize();
        return Quat::from_axis_angle(axis, PI);
    }
    let c = f.cross(t);
    Quat::from_xyzw(c.x, c.y, c.z, 1.0 + dot).normalize()
}

fn parse_mjcf_pose(attrs: &AttrMap, compiler: &Compiler) -> Result<Pose, String> {
    let pos = match attrs.get("pos") {
        Some(s) => parse_vec3(s)?,
        None => Vec3::ZERO,
    };
    let rot = if let Some(s) = attrs.get("quat") {
        let f = parse_f64_list(s)?;
        if f.len() != 4 {
            return Err(format!("expected 4 floats in MJCF quat '{s}'"));
        }
        // MJCF stores quaternions in scalar-first `w x y z` order.
        Quat::from_xyzw(f[1], f[2], f[3], f[0]).normalize()
    } else if let Some(s) = attrs.get("axisangle") {
        let f = parse_f64_list(s)?;
        if f.len() != 4 {
            return Err(format!("expected 4 floats in MJCF axisangle '{s}'"));
        }
        let axis = Vec3::new(f[0], f[1], f[2])
            .try_normalize()
            .ok_or_else(|| format!("zero axis in MJCF axisangle '{s}'"))?;
        Quat::from_axis_angle(axis, f[3] * compiler.angle_scale)
    } else if let Some(s) = attrs.get("euler") {
        let f = parse_f64_list(s)?;
        if f.len() != 3 {
            return Err(format!("expected 3 floats in MJCF euler '{s}'"));
        }
        let chars: Vec<char> = compiler.eulerseq.chars().collect();
        let q0 = quat_from_axis_char(chars[0], f[0] * compiler.angle_scale);
        let q1 = quat_from_axis_char(chars[1], f[1] * compiler.angle_scale);
        let q2 = quat_from_axis_char(chars[2], f[2] * compiler.angle_scale);
        if chars[0].is_ascii_lowercase() {
            (q0 * q1 * q2).normalize()
        } else {
            (q2 * q1 * q0).normalize()
        }
    } else if let Some(s) = attrs.get("xyaxes") {
        let f = parse_f64_list(s)?;
        if f.len() != 6 {
            return Err(format!("expected 6 floats in MJCF xyaxes '{s}'"));
        }
        let x = Vec3::new(f[0], f[1], f[2])
            .try_normalize()
            .ok_or_else(|| format!("zero X axis in MJCF xyaxes '{s}'"))?;
        let y_raw = Vec3::new(f[3], f[4], f[5]);
        let z = x
            .cross(y_raw)
            .try_normalize()
            .ok_or_else(|| format!("collinear vectors in MJCF xyaxes '{s}'"))?;
        let y = z.cross(x).normalize();
        Quat::from_mat3(&Mat3::from_cols(x, y, z))
    } else if let Some(s) = attrs.get("zaxis") {
        let v = parse_vec3(s)?
            .try_normalize()
            .ok_or_else(|| format!("zero vector in MJCF zaxis '{s}'"))?;
        quat_from_two_unit_vecs(Vec3::Z, v)
    } else {
        Quat::IDENTITY
    };
    Ok(Pose::new(pos, rot))
}

fn parse_two_f64(s: &str, context: &str) -> Result<(f64, f64), String> {
    let f = parse_f64_list(s)?;
    if f.len() != 2 {
        return Err(format!("expected 2 floats in {context}: '{s}'"));
    }
    Ok((f[0], f[1]))
}

fn parse_float(attrs: &AttrMap, key: &str) -> Result<Option<f64>, String> {
    match attrs.get(key) {
        Some(s) => s
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("invalid float '{s}' for '{key}'")),
        None => Ok(None),
    }
}

impl<'a> MjcfBuilder<'a> {
    fn next_name(&mut self, prefix: &str) -> String {
        let idx = self.unnamed_counter;
        self.unnamed_counter += 1;
        format!("{prefix}_{idx}")
    }

    fn resolve_material(&mut self, attrs: &AttrMap) -> Result<MaterialId, String> {
        let base = self.world.material(self.default_material).clone();
        let friction = match attrs.get("friction") {
            Some(s) => {
                let vals = parse_f64_list(s)?;
                vals.first().copied().unwrap_or(base.friction).max(0.0)
            }
            None => base.friction,
        };
        let density = parse_float(attrs, "density")?
            .unwrap_or(base.density)
            .max(1e-3);
        if friction.to_bits() == base.friction.to_bits()
            && density.to_bits() == base.density.to_bits()
        {
            return Ok(self.default_material);
        }
        let key = (friction.to_bits(), density.to_bits());
        if let Some(&id) = self.material_cache.get(&key) {
            return Ok(id);
        }
        let mut mat = base;
        mat.name = format!("{}#mjcf", mat.name);
        mat.friction = friction;
        mat.density = density;
        let id = self.world.add_material(mat);
        self.material_cache.insert(key, id);
        Ok(id)
    }

    fn parse_geom(
        &mut self,
        node: &XmlNode,
        active_class: Option<&str>,
    ) -> Result<GeomSpec, String> {
        let attrs = resolved_attrs(&self.defaults, node, active_class);
        let name = attrs.get("name").cloned();
        let kind = attrs.get("type").map(String::as_str).unwrap_or("sphere");
        let contype = parse_float(&attrs, "contype")?.unwrap_or(1.0);
        let conaffinity = parse_float(&attrs, "conaffinity")?.unwrap_or(1.0);
        let is_collision = contype != 0.0 || conaffinity != 0.0;

        let material = self.resolve_material(&attrs)?;
        let mass_override = parse_float(&attrs, "mass")?.filter(|&m| m > 0.0);
        let size = match attrs.get("size") {
            Some(s) => parse_f64_list(s)?,
            None => Vec::new(),
        };

        let mut local_pose = parse_mjcf_pose(&attrs, &self.compiler)?;
        let z_to_y = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);

        let shape = if let Some(fromto_str) = attrs.get("fromto") {
            let f = parse_f64_list(fromto_str)?;
            if f.len() != 6 {
                return Err(format!("expected 6 floats in MJCF fromto '{fromto_str}'"));
            }
            let p1 = Vec3::new(f[0], f[1], f[2]);
            let p2 = Vec3::new(f[3], f[4], f[5]);
            let diff = p2 - p1;
            let len = diff.length();
            let dir = if len > 1e-12 { diff / len } else { Vec3::Y };
            local_pose = Pose::new(0.5 * (p1 + p2), quat_from_two_unit_vecs(Vec3::Y, dir));
            let radius = *size
                .first()
                .ok_or("MJCF fromto capsule/cylinder missing size radius")?;
            match kind {
                "cylinder" => Shape::Capsule {
                    radius,
                    half_length: (0.5 * len - radius).max(0.0),
                },
                "box" => {
                    let hy = 0.5 * len;
                    let hx = radius;
                    let hz = size.get(1).copied().unwrap_or(radius);
                    Shape::Cuboid {
                        half: Vec3::new(hx, hy, hz),
                    }
                }
                _ => Shape::Capsule {
                    radius,
                    half_length: 0.5 * len,
                },
            }
        } else {
            match kind {
                "plane" => Shape::HalfSpace {
                    normal: Vec3::Z,
                    offset: 0.0,
                },
                "sphere" => {
                    let radius = size.first().copied().unwrap_or(0.05).max(1e-5);
                    Shape::Sphere { radius }
                }
                "box" => {
                    if size.len() < 3 {
                        return Err("MJCF box <geom> requires 3 size half-extents".into());
                    }
                    Shape::Cuboid {
                        half: Vec3::new(size[0], size[1], size[2]),
                    }
                }
                "capsule" => {
                    let radius = size.first().copied().unwrap_or(0.05).max(1e-5);
                    let half_length = size.get(1).copied().unwrap_or(0.0).max(0.0);
                    local_pose.rotation = (local_pose.rotation * z_to_y).normalize();
                    Shape::Capsule {
                        radius,
                        half_length,
                    }
                }
                "cylinder" => {
                    let radius = size.first().copied().unwrap_or(0.05).max(1e-5);
                    let half_len = size.get(1).copied().unwrap_or(radius).max(0.0);
                    local_pose.rotation = (local_pose.rotation * z_to_y).normalize();
                    Shape::Capsule {
                        radius,
                        half_length: (half_len - radius).max(0.0),
                    }
                }
                "ellipsoid" => {
                    let radius = if size.len() >= 3 {
                        (size[0] * size[1] * size[2]).max(1e-12).cbrt()
                    } else {
                        size.first().copied().unwrap_or(0.05).max(1e-5)
                    };
                    Shape::Sphere { radius }
                }
                "mesh" => {
                    let radius = size.first().copied().unwrap_or(0.05).max(1e-5);
                    Shape::Sphere { radius }
                }
                other => return Err(format!("unsupported MJCF geom type '{other}'")),
            }
        };

        Ok(GeomSpec {
            name,
            shape,
            local_pose,
            material,
            mass_override,
            is_collision,
        })
    }

    fn parse_inertial(&self, body_node: &XmlNode) -> Result<Option<InertialSpec>, String> {
        let Some(inode) = body_node.children.iter().find(|c| c.tag == "inertial") else {
            return Ok(None);
        };
        let attrs = resolved_attrs(&self.defaults, inode, None);
        let local_pose = parse_mjcf_pose(&attrs, &self.compiler)?;
        let mass = parse_float(&attrs, "mass")?;
        let inertia = if let Some(s) = attrs.get("diaginertia") {
            let v = parse_vec3(s)?;
            Some(Mat3::from_diagonal(v))
        } else if let Some(s) = attrs.get("fullinertia") {
            let f = parse_f64_list(s)?;
            if f.len() != 6 {
                return Err(format!("expected 6 floats in MJCF fullinertia '{s}'"));
            }
            Some(Mat3::from_rows(
                Vec3::new(f[0], f[3], f[4]),
                Vec3::new(f[3], f[1], f[5]),
                Vec3::new(f[4], f[5], f[2]),
            ))
        } else {
            None
        };
        Ok(Some(InertialSpec {
            local_pose,
            mass,
            inertia,
        }))
    }

    fn parse_joint_spec(
        &self,
        node: &XmlNode,
        active_class: Option<&str>,
    ) -> Result<JointSpec, String> {
        if node.tag == "freejoint" {
            return Ok(JointSpec {
                name: node.attrs.get("name").cloned(),
                kind: "free".into(),
                pos: Vec3::ZERO,
                axis: Vec3::Z,
                limits: None,
                stiffness: 0.0,
                damping: 0.0,
                springref: 0.0,
            });
        }
        let attrs = resolved_attrs(&self.defaults, node, active_class);
        let kind = attrs.get("type").cloned().unwrap_or_else(|| "hinge".into());
        let pos = match attrs.get("pos") {
            Some(s) => parse_vec3(s)?,
            None => Vec3::ZERO,
        };
        let axis = match attrs.get("axis") {
            Some(s) => parse_vec3(s)?.try_normalize().unwrap_or(Vec3::Z),
            None => Vec3::Z,
        };
        let raw_range = match attrs.get("range") {
            Some(s) => {
                let (lo, hi) = parse_two_f64(s, "MJCF joint range")?;
                let scale = if kind == "hinge" {
                    self.compiler.angle_scale
                } else {
                    1.0
                };
                let (lo, hi) = (lo * scale, hi * scale);
                if hi >= lo {
                    Some((lo, hi))
                } else {
                    None
                }
            }
            None => None,
        };
        let limits = match attrs.get("limited").map(String::as_str) {
            Some("false") => None,
            Some("true") => raw_range,
            _ => {
                if self.compiler.autolimits {
                    raw_range
                } else {
                    None
                }
            }
        };
        let stiffness = parse_float(&attrs, "stiffness")?.unwrap_or(0.0).max(0.0);
        let damping = parse_float(&attrs, "damping")?.unwrap_or(0.0).max(0.0);
        let springref_raw = parse_float(&attrs, "springref")?.unwrap_or(0.0);
        let springref = if kind == "hinge" {
            springref_raw * self.compiler.angle_scale
        } else {
            springref_raw
        };
        Ok(JointSpec {
            name: attrs.get("name").cloned(),
            kind,
            pos,
            axis,
            limits,
            stiffness,
            damping,
            springref,
        })
    }

    fn record_site(
        &mut self,
        node: &XmlNode,
        active_class: Option<&str>,
        owner: Option<(ObjectId, Pose)>,
        body_world_pose: Pose,
    ) -> Result<(), String> {
        let attrs = resolved_attrs(&self.defaults, node, active_class);
        let Some(name) = attrs.get("name").cloned() else {
            return Ok(());
        };
        let site_local = parse_mjcf_pose(&attrs, &self.compiler)?;
        let site_world = compose_pose(body_world_pose, site_local);
        let size = match attrs.get("size") {
            Some(s) => parse_f64_list(s)?,
            None => Vec::new(),
        };
        let radius = size.first().copied().unwrap_or(0.02).max(1e-4);
        match owner {
            Some((obj_id, obj_world_pose)) => {
                let obj_inv = inverse_pose(obj_world_pose);
                let local = obj_inv.transform_point(site_world.position);
                let local_normal = obj_inv
                    .rotation
                    .rotate(site_world.rotation.rotate(Vec3::Z))
                    .normalize();
                self.sites.insert(
                    name.clone(),
                    LinkPoint::Body {
                        object: obj_id,
                        local,
                    },
                );
                self.site_normals.insert(name, (local_normal, radius));
            }
            None => {
                self.sites
                    .insert(name.clone(), LinkPoint::World(site_world.position));
                self.site_normals.insert(name, (Vec3::Z, radius));
            }
        }
        Ok(())
    }

    fn spawn_body(
        &mut self,
        node: &XmlNode,
        parent_link: Option<(ObjectId, Pose)>,
        parent_all_ids: &[ObjectId],
        parent_body_world: Pose,
        inherited_class: Option<&str>,
    ) -> Result<(), String> {
        let body_pose_local = parse_mjcf_pose(&node.attrs, &self.compiler)?;
        let body_world_pose = compose_pose(parent_body_world, body_pose_local);
        let active_class = node
            .attrs
            .get("childclass")
            .map(String::as_str)
            .or(inherited_class);
        let body_name = node
            .attrs
            .get("name")
            .cloned()
            .unwrap_or_else(|| self.next_name("body"));

        let mut all_geoms = Vec::new();
        for child in &node.children {
            if child.tag == "geom" {
                all_geoms.push(self.parse_geom(child, active_class)?);
            }
        }
        let mut active_geoms: Vec<GeomSpec> = all_geoms
            .iter()
            .filter(|g| g.is_collision)
            .cloned()
            .collect();
        if active_geoms.is_empty() && !all_geoms.is_empty() {
            active_geoms.push(all_geoms[0].clone());
        }

        let inertial = self.parse_inertial(node)?;
        let mut joint_specs = Vec::new();
        for child in &node.children {
            if child.tag == "joint" || child.tag == "freejoint" {
                joint_specs.push(self.parse_joint_spec(child, active_class)?);
            }
        }

        let is_fixed_body = (parent_link.is_none() && joint_specs.is_empty())
            || inertial.as_ref().and_then(|i| i.mass) == Some(0.0);

        let primary_local_pose = active_geoms
            .first()
            .map(|g| g.local_pose)
            .or_else(|| inertial.as_ref().map(|i| i.local_pose))
            .unwrap_or(Pose::IDENTITY);
        let primary_world_pose = compose_pose(body_world_pose, primary_local_pose);

        let mut spawned_ids = Vec::new();
        let primary_id = if active_geoms.is_empty() {
            let shape = Shape::Sphere { radius: 0.02 };
            if is_fixed_body {
                self.world
                    .add_fixed(&body_name, self.default_material, shape, primary_world_pose)
            } else {
                let id = self.world.add_rigid(
                    &body_name,
                    self.default_material,
                    shape,
                    primary_world_pose,
                );
                self.apply_inertial(id, primary_local_pose, inertial.as_ref(), None);
                id
            }
        } else {
            let g0 = &active_geoms[0];
            let g_world = compose_pose(body_world_pose, g0.local_pose);
            let id = if is_fixed_body || !g0.shape.is_bounded() {
                self.world
                    .add_fixed(&body_name, g0.material, g0.shape, g_world)
            } else {
                let id = self
                    .world
                    .add_rigid(&body_name, g0.material, g0.shape, g_world);
                let extra_share = if active_geoms.len() > 1 {
                    let vols: Vec<f64> = active_geoms
                        .iter()
                        .map(|g| {
                            g.mass_override.unwrap_or_else(|| {
                                g.shape.unit_mass().map_or(1.0, |u| {
                                    u.volume * self.world.material(g.material).density
                                })
                            })
                        })
                        .collect();
                    let sum: f64 = vols.iter().sum();
                    if sum > 0.0 {
                        Some(vols[0] / sum)
                    } else {
                        None
                    }
                } else {
                    None
                };
                self.apply_inertial(
                    id,
                    g0.local_pose,
                    inertial.as_ref(),
                    g0.mass_override.or_else(|| {
                        extra_share
                            .and_then(|s| inertial.as_ref().and_then(|i| i.mass.map(|m| m * s)))
                    }),
                );
                id
            };
            if let Some(ref gname) = g0.name {
                self.geoms_by_name.insert(gname.clone(), id);
            }
            id
        };
        if self.root_object.is_none() {
            self.root_object = Some(primary_id);
        }
        spawned_ids.push(primary_id);
        self.links_by_name.insert(body_name.clone(), primary_id);

        // Spawn compound extra geoms and weld them rigidly onto the primary body.
        for (idx, g) in active_geoms.iter().enumerate().skip(1) {
            let aux_name = g
                .name
                .clone()
                .unwrap_or_else(|| format!("{body_name}#geom{idx}"));
            let g_world = compose_pose(body_world_pose, g.local_pose);
            let aux_id = if is_fixed_body || !g.shape.is_bounded() {
                self.world
                    .add_fixed(&aux_name, g.material, g.shape, g_world)
            } else {
                let id = self
                    .world
                    .add_rigid(&aux_name, g.material, g.shape, g_world);
                if let Some(m) = g.mass_override {
                    self.rescale_rigid_mass(id, m);
                }
                let p_inv = inverse_pose(primary_world_pose);
                let g_inv = inverse_pose(g_world);
                let rel_rot =
                    (primary_world_pose.rotation.conjugate() * g_world.rotation).normalize();
                let local_p = p_inv.transform_point(g_world.position);
                let m_scale = self.world.object(id).mass().clamp(0.1, 100.0);
                self.world.add_weld(
                    id,
                    Some(primary_id),
                    g_inv.transform_point(g_world.position),
                    local_p,
                    rel_rot,
                    1e6 * m_scale,
                    5e4 * m_scale,
                );
                id
            };
            for &prev in &spawned_ids {
                self.world.exclude_contact(prev, aux_id);
            }
            if let Some((pid, _)) = parent_link {
                self.world.exclude_contact(pid, aux_id);
            }
            if let Some(ref gname) = g.name {
                self.geoms_by_name.insert(gname.clone(), aux_id);
            }
            spawned_ids.push(aux_id);
        }
        self.body_objects
            .insert(body_name.clone(), spawned_ids.clone());

        // Spawn body sites (used for tendons and tactile/IMU/wrench probes).
        for child in &node.children {
            if child.tag == "site" {
                self.record_site(
                    child,
                    active_class,
                    Some((primary_id, primary_world_pose)),
                    body_world_pose,
                )?;
            }
        }

        // Connect primary body to parent (or world) using parsed MJCF joint specs.
        if !is_fixed_body {
            self.connect_body_joints(
                primary_id,
                primary_world_pose,
                parent_link,
                body_world_pose,
                &joint_specs,
            )?;
            let is_free = joint_specs.iter().any(|s| s.kind == "free");
            if !is_free {
                for &p_oid in parent_all_ids {
                    for &c_oid in &spawned_ids {
                        self.world.exclude_contact(p_oid, c_oid);
                    }
                }
            }
        }

        // Recurse into child bodies.
        for child in &node.children {
            if child.tag == "body" {
                self.spawn_body(
                    child,
                    Some((primary_id, primary_world_pose)),
                    &spawned_ids,
                    body_world_pose,
                    active_class,
                )?;
            }
        }
        Ok(())
    }

    fn rescale_rigid_mass(&mut self, id: ObjectId, mass: f64) {
        if let Representation::Rigid(r) = &mut self.world.object_mut(id).representation {
            if r.mass > 0.0 && mass > 0.0 {
                let ratio = mass / r.mass;
                r.mass = mass;
                r.inertia *= ratio;
            }
        }
    }

    fn apply_inertial(
        &mut self,
        id: ObjectId,
        geom_local: Pose,
        inertial: Option<&InertialSpec>,
        mass_override: Option<f64>,
    ) {
        let Some(iner_spec) = inertial else {
            if let Some(m) = mass_override {
                self.rescale_rigid_mass(id, m);
            }
            return;
        };
        let Representation::Rigid(r) = &mut self.world.object_mut(id).representation else {
            return;
        };
        let target_mass = mass_override.or(iner_spec.mass).filter(|&m| m > 0.0);
        if let Some(m) = target_mass {
            if r.mass > 0.0 {
                r.inertia *= m / r.mass;
            }
            r.mass = m;
        }
        if let Some(i_mat) = iner_spec.inertia {
            if i_mat.diagonal().min_element() > 0.0 {
                let r_rel = (geom_local.rotation.conjugate() * iner_spec.local_pose.rotation)
                    .normalize()
                    .to_mat3();
                let mut i_rot = r_rel * i_mat * r_rel.transpose();
                let d = geom_local
                    .rotation
                    .conjugate()
                    .rotate(iner_spec.local_pose.position - geom_local.position);
                if d.length_squared() > 1e-10 {
                    i_rot += (Mat3::IDENTITY * d.length_squared() - Mat3::outer(d, d)) * r.mass;
                }
                r.inertia = i_rot;
            }
        }
    }

    fn connect_body_joints(
        &mut self,
        child_id: ObjectId,
        child_pose: Pose,
        parent_link: Option<(ObjectId, Pose)>,
        body_world_pose: Pose,
        specs: &[JointSpec],
    ) -> Result<(), String> {
        let parent_id = parent_link.map(|(id, _)| id);
        let parent_pose = parent_link.map_or(Pose::IDENTITY, |(_, p)| p);
        let c_inv = inverse_pose(child_pose);
        let p_inv = inverse_pose(parent_pose);
        let rel_rot = (parent_pose.rotation.conjugate() * child_pose.rotation).normalize();
        let m_scale = self.world.object(child_id).mass().clamp(0.1, 100.0);
        let k_lin = 1e6 * m_scale;
        let k_ang = 5e4 * m_scale;

        if specs.is_empty() {
            let anchor_w = body_world_pose.position;
            let local_a = c_inv.transform_point(anchor_w);
            let local_b = if parent_id.is_some() {
                p_inv.transform_point(anchor_w)
            } else {
                anchor_w
            };
            self.world
                .add_weld(child_id, parent_id, local_a, local_b, rel_rot, k_lin, k_ang);
            return Ok(());
        }

        if specs.len() == 1 && specs[0].kind == "free" {
            return Ok(());
        }
        if specs.iter().any(|s| s.kind == "free") {
            return Err("MJCF <freejoint> cannot be combined with other joints on one body".into());
        }

        let record_joint = |this: &mut Self, j: &JointSpec, lid: LinkId| {
            this.body_parent_joint.entry(child_id).or_insert(lid);
            if let Some(ref jname) = j.name {
                this.joints_by_name.insert(jname.clone(), lid);
                this.joint_body.insert(jname.clone(), child_id);
                this.joint_specs.insert(jname.clone(), j.clone());
            }
        };

        if specs.len() == 1 {
            let j = &specs[0];
            let anchor_w = body_world_pose.transform_point(j.pos);
            let local_a = c_inv.transform_point(anchor_w);
            let local_b = if parent_id.is_some() {
                p_inv.transform_point(anchor_w)
            } else {
                anchor_w
            };
            let axis_w = body_world_pose.rotation.rotate(j.axis).normalize();
            let axis_a = c_inv.rotation.rotate(axis_w).normalize();
            let axis_b = if parent_id.is_some() {
                p_inv.rotation.rotate(axis_w).normalize()
            } else {
                axis_w
            };
            let drive = JointDrive::pd(j.springref, j.stiffness, j.damping);
            let link_id = match j.kind.as_str() {
                "hinge" => self.world.add_hinge(
                    child_id, parent_id, local_a, local_b, axis_a, rel_rot, k_lin, k_ang, j.limits,
                    drive,
                ),
                "slide" => self.world.add_slider(
                    child_id, parent_id, local_a, local_b, axis_b, rel_rot, k_lin, k_ang, j.limits,
                    drive,
                ),
                "ball" => self.world.add_ball_joint(
                    LinkPoint::Body {
                        object: child_id,
                        local: local_a,
                    },
                    match parent_id {
                        Some(pid) => LinkPoint::Body {
                            object: pid,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    },
                    k_lin,
                ),
                other => return Err(format!("unsupported MJCF joint type '{other}'")),
            };
            record_joint(self, j, link_id);
            return Ok(());
        }

        if specs.len() == 2 {
            let (j0, j1) = (&specs[0], &specs[1]);
            let anchor_w = body_world_pose.transform_point(j0.pos);
            let local_a = c_inv.transform_point(anchor_w);
            let local_b = if parent_id.is_some() {
                p_inv.transform_point(anchor_w)
            } else {
                anchor_w
            };
            let a0_w = body_world_pose.rotation.rotate(j0.axis).normalize();
            let a1_w = body_world_pose.rotation.rotate(j1.axis).normalize();
            if (j0.kind == "hinge" && j1.kind == "slide")
                || (j0.kind == "slide" && j1.kind == "hinge")
            {
                let (h, s) = if j0.kind == "hinge" {
                    (j0, j1)
                } else {
                    (j1, j0)
                };
                if a0_w.dot(a1_w).abs() > 0.99 {
                    let axis_a = c_inv
                        .rotation
                        .rotate(body_world_pose.rotation.rotate(h.axis))
                        .normalize();
                    let link_id = self.world.add_cylindrical(
                        child_id,
                        parent_id,
                        local_a,
                        local_b,
                        axis_a,
                        rel_rot,
                        k_lin,
                        k_ang,
                        s.limits,
                        h.limits,
                        JointDrive::pd(s.springref, s.stiffness, s.damping),
                        JointDrive::pd(h.springref, h.stiffness, h.damping),
                    );
                    record_joint(self, s, link_id);
                    record_joint(self, h, link_id);
                    return Ok(());
                }
            } else if j0.kind == "hinge" && j1.kind == "hinge" && a0_w.dot(a1_w).abs() < 0.05 {
                let axis_a = c_inv.rotation.rotate(a1_w).normalize();
                let axis_b = if parent_id.is_some() {
                    p_inv.rotation.rotate(a0_w).normalize()
                } else {
                    a0_w
                };
                let link_id = self.world.add_universal(
                    child_id, parent_id, local_a, local_b, axis_a, axis_b, k_lin, k_ang,
                );
                record_joint(self, j0, link_id);
                record_joint(self, j1, link_id);
                return Ok(());
            } else if j0.kind == "slide" && j1.kind == "slide" && a0_w.dot(a1_w).abs() < 0.05 {
                let n_w = a0_w.cross(a1_w).normalize();
                let normal_a = c_inv.rotation.rotate(n_w).normalize();
                let normal_b = if parent_id.is_some() {
                    p_inv.rotation.rotate(n_w).normalize()
                } else {
                    n_w
                };
                let link_id = self.world.add_planar(
                    child_id, parent_id, local_a, local_b, normal_a, normal_b, k_lin, k_ang,
                );
                record_joint(self, j0, link_id);
                record_joint(self, j1, link_id);
                return Ok(());
            }
        }

        if specs.len() == 3 {
            let slides: Vec<&JointSpec> = specs.iter().filter(|s| s.kind == "slide").collect();
            let hinges: Vec<&JointSpec> = specs.iter().filter(|s| s.kind == "hinge").collect();
            if slides.len() == 2 && hinges.len() == 1 {
                let s0_w = body_world_pose.rotation.rotate(slides[0].axis).normalize();
                let s1_w = body_world_pose.rotation.rotate(slides[1].axis).normalize();
                let h_w = body_world_pose.rotation.rotate(hinges[0].axis).normalize();
                let n_w = s0_w.cross(s1_w).try_normalize();
                if let Some(n_w) = n_w {
                    if s0_w.dot(s1_w).abs() < 0.05 && n_w.dot(h_w).abs() > 0.99 {
                        let anchor_w = body_world_pose.transform_point(hinges[0].pos);
                        let local_a = c_inv.transform_point(anchor_w);
                        let local_b = if parent_id.is_some() {
                            p_inv.transform_point(anchor_w)
                        } else {
                            anchor_w
                        };
                        let normal_a = c_inv.rotation.rotate(n_w).normalize();
                        let normal_b = if parent_id.is_some() {
                            p_inv.rotation.rotate(n_w).normalize()
                        } else {
                            n_w
                        };
                        let link_id = self.world.add_planar(
                            child_id, parent_id, local_a, local_b, normal_a, normal_b, k_lin, k_ang,
                        );
                        for j in specs {
                            record_joint(self, j, link_id);
                        }
                        return Ok(());
                    }
                }
            }
        }

        Err(format!(
            "unsupported combination of {} MJCF joints on one body",
            specs.len()
        ))
    }

    fn site_world_pos(&self, pt: LinkPoint) -> Vec3 {
        match pt {
            LinkPoint::World(p) => p,
            LinkPoint::Body { object, local } => match &self.world.object(object).representation {
                Representation::Fixed(f) => f.pose.transform_point(local),
                Representation::Rigid(r) => r.pose.transform_point(local),
                Representation::Solid(s) => s.nodes.position[0],
            },
            LinkPoint::Node { object, node } => match &self.world.object(object).representation {
                Representation::Solid(s) => s.nodes.position[node as usize],
                _ => Vec3::ZERO,
            },
        }
    }

    fn process_contacts(&mut self, root: &XmlNode) {
        for section in root.children.iter().filter(|c| c.tag == "contact") {
            for child in section.children.iter().filter(|c| c.tag == "exclude") {
                let (Some(b1), Some(b2)) = (child.attrs.get("body1"), child.attrs.get("body2"))
                else {
                    continue;
                };
                let objs1 = self.body_objects.get(b1).cloned().unwrap_or_default();
                let objs2 = self.body_objects.get(b2).cloned().unwrap_or_default();
                for &o1 in &objs1 {
                    for &o2 in &objs2 {
                        self.world.exclude_contact(o1, o2);
                    }
                }
            }
        }
    }

    fn process_tendons(&mut self, root: &XmlNode) -> Result<(), String> {
        for section in root.children.iter().filter(|c| c.tag == "tendon") {
            for child in &section.children {
                if child.tag != "spatial" {
                    continue;
                }
                let attrs = resolved_attrs(&self.defaults, child, None);
                let name = attrs
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| self.next_name("tendon"));
                let mut waypoints = Vec::new();
                for wp in &child.children {
                    if wp.tag == "site" {
                        let sname = wp
                            .attrs
                            .get("site")
                            .ok_or_else(|| format!("tendon '{name}' <site> missing site attr"))?;
                        let pt = self.sites.get(sname).copied().ok_or_else(|| {
                            format!("tendon '{name}' references unknown site '{sname}'")
                        })?;
                        waypoints.push(pt);
                    }
                }
                if waypoints.len() < 2 {
                    return Err(format!(
                        "MJCF spatial tendon '{name}' requires >= 2 <site>s"
                    ));
                }
                let initial_len: f64 = waypoints
                    .windows(2)
                    .map(|w| (self.site_world_pos(w[1]) - self.site_world_pos(w[0])).length())
                    .sum();
                let rest_len = match attrs.get("springlength") {
                    Some(s) => {
                        let f = parse_f64_list(s)?;
                        f.first()
                            .copied()
                            .filter(|&v| v >= 0.0)
                            .unwrap_or(initial_len)
                    }
                    None => initial_len,
                };
                let stiffness = parse_float(&attrs, "stiffness")?.unwrap_or(0.0).max(0.0);
                let damping = parse_float(&attrs, "damping")?.unwrap_or(0.0).max(0.0);
                let limited = attrs.get("limited").map(String::as_str) == Some("true");
                let target = if limited {
                    if let Some(s) = attrs.get("range") {
                        let (_, hi) = parse_two_f64(s, "MJCF tendon range")?;
                        hi
                    } else {
                        rest_len
                    }
                } else {
                    rest_len
                };
                let lid = self.world.add_tendon(
                    waypoints,
                    stiffness,
                    damping,
                    attrs.contains_key("springlength") || limited,
                    JointDrive::pd(target, 0.0, 0.0),
                );
                self.tendons_by_name.insert(name.clone(), lid);
                self.joints_by_name.insert(name, lid);
            }
        }
        Ok(())
    }

    fn process_actuators(&mut self, root: &XmlNode) -> Result<(), String> {
        for section in root.children.iter().filter(|c| c.tag == "actuator") {
            for child in &section.children {
                if !matches!(
                    child.tag.as_str(),
                    "position" | "velocity" | "motor" | "general"
                ) {
                    continue;
                }
                let attrs = resolved_attrs(&self.defaults, child, None);
                let name = attrs
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| self.next_name(&child.tag));
                let link_id = if let Some(jname) = attrs.get("joint") {
                    *self.joints_by_name.get(jname).ok_or_else(|| {
                        format!("actuator '{name}' references unknown joint '{jname}'")
                    })?
                } else if let Some(tname) = attrs.get("tendon") {
                    *self.tendons_by_name.get(tname).ok_or_else(|| {
                        format!("actuator '{name}' references unknown tendon '{tname}'")
                    })?
                } else {
                    return Err(format!(
                        "MJCF actuator '{name}' must specify joint or tendon"
                    ));
                };

                let passive = attrs
                    .get("joint")
                    .and_then(|j| self.joint_specs.get(j))
                    .cloned();
                let mut drive = match self.world.link(link_id) {
                    crate::Link::Hinge { drive, .. }
                    | crate::Link::Slider { drive, .. }
                    | crate::Link::Tendon { drive, .. } => *drive,
                    crate::Link::Cylindrical { linear_drive, .. } => *linear_drive,
                    _ => JointDrive::NONE,
                };

                let max_force = if let Some(s) = attrs.get("forcerange") {
                    let (lo, hi) = parse_two_f64(s, "MJCF actuator forcerange")?;
                    Some(lo.abs().max(hi.abs()))
                } else if child.tag == "motor" {
                    if let Some(s) = attrs.get("ctrlrange") {
                        let (lo, hi) = parse_two_f64(s, "MJCF motor ctrlrange")?;
                        let gear = parse_float(&attrs, "gear")?.unwrap_or(1.0).abs();
                        Some(lo.abs().max(hi.abs()) * gear)
                    } else {
                        None
                    }
                } else {
                    None
                };

                match child.tag.as_str() {
                    "position" => {
                        let kp = parse_float(&attrs, "kp")?.unwrap_or(100.0);
                        let kv = parse_float(&attrs, "kv")?;
                        drive.stiffness = kp;
                        if let Some(kv) = kv {
                            drive.damping = passive.as_ref().map_or(0.0, |p| p.damping) + kv;
                        }
                        if let Some(mf) = max_force {
                            drive.max_force = Some(mf);
                        }
                    }
                    "velocity" => {
                        let kv = parse_float(&attrs, "kv")?.unwrap_or(10.0);
                        drive.damping = passive.as_ref().map_or(0.0, |p| p.damping) + kv;
                        if let Some(mf) = max_force {
                            drive.max_force = Some(mf);
                        }
                    }
                    "motor" => {
                        if let Some(mf) = max_force {
                            drive.max_force = Some(mf);
                        }
                    }
                    "general" => {
                        let gain = attrs
                            .get("gainprm")
                            .and_then(|s| parse_f64_list(s).ok())
                            .and_then(|v| v.first().copied());
                        let bias = attrs
                            .get("biasprm")
                            .and_then(|s| parse_f64_list(s).ok())
                            .unwrap_or_default();
                        if let Some(g) = gain {
                            if attrs.get("biastype").map(String::as_str) == Some("affine")
                                || bias.len() >= 2
                            {
                                let kp = bias.get(1).copied().map_or(g, |b| (-b).max(0.0));
                                let kv = bias.get(2).copied().map_or(0.0, |b| (-b).max(0.0));
                                drive.stiffness = kp.max(0.0);
                                drive.damping = passive.as_ref().map_or(0.0, |p| p.damping) + kv;
                            } else {
                                drive.stiffness = g.abs();
                            }
                        }
                        if let Some(mf) = max_force {
                            drive.max_force = Some(mf);
                        }
                    }
                    _ => {}
                }

                match self.world.link_mut(link_id) {
                    crate::Link::Hinge { drive: d, .. }
                    | crate::Link::Slider { drive: d, .. }
                    | crate::Link::Tendon { drive: d, .. } => *d = drive,
                    crate::Link::Cylindrical {
                        linear_drive,
                        angular_drive,
                        ..
                    } => {
                        if passive.as_ref().is_some_and(|p| p.kind == "hinge") {
                            *angular_drive = drive;
                        } else {
                            *linear_drive = drive;
                        }
                    }
                    _ => {}
                }
                self.actuators_by_name.insert(name, link_id);
            }
        }
        Ok(())
    }

    fn process_sensors(&mut self, root: &XmlNode) -> Result<(), String> {
        for section in root.children.iter().filter(|c| c.tag == "sensor") {
            for child in &section.children {
                let name = child
                    .attrs
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| self.next_name(&child.tag));
                let probe = match child.tag.as_str() {
                    "jointpos" | "jointvel" => {
                        let jname = child.attrs.get("joint").ok_or_else(|| {
                            format!("sensor '{name}' <{}> missing joint", child.tag)
                        })?;
                        let link = *self.joints_by_name.get(jname).ok_or_else(|| {
                            format!("sensor '{name}' references unknown joint '{jname}'")
                        })?;
                        Probe::JointState { link }
                    }
                    "tendonpos" | "tendonvel" => {
                        let tname = child.attrs.get("tendon").ok_or_else(|| {
                            format!("sensor '{name}' <{}> missing tendon", child.tag)
                        })?;
                        let link = *self.tendons_by_name.get(tname).ok_or_else(|| {
                            format!("sensor '{name}' references unknown tendon '{tname}'")
                        })?;
                        Probe::JointState { link }
                    }
                    "touch" => {
                        let sname = child
                            .attrs
                            .get("site")
                            .ok_or_else(|| format!("sensor '{name}' <touch> missing site"))?;
                        let LinkPoint::Body { object, local } =
                            self.sites.get(sname).copied().ok_or_else(|| {
                                format!("sensor '{name}' references unknown body site '{sname}'")
                            })?
                        else {
                            return Err(format!("touch sensor '{name}' requires a body site"));
                        };
                        let (local_normal, radius) = self
                            .site_normals
                            .get(sname)
                            .copied()
                            .unwrap_or((Vec3::Z, 0.02));
                        Probe::Tactile {
                            object,
                            local_origin: local,
                            local_normal,
                            radius,
                        }
                    }
                    "accelerometer" | "gyro" => {
                        let sname = child.attrs.get("site").ok_or_else(|| {
                            format!("sensor '{name}' <{}> missing site", child.tag)
                        })?;
                        let LinkPoint::Body { object, local } =
                            self.sites.get(sname).copied().ok_or_else(|| {
                                format!("sensor '{name}' references unknown body site '{sname}'")
                            })?
                        else {
                            return Err(format!("IMU sensor '{name}' requires a body site"));
                        };
                        Probe::Imu { object, local }
                    }
                    "force" | "torque" => {
                        let sname = child.attrs.get("site").ok_or_else(|| {
                            format!("sensor '{name}' <{}> missing site", child.tag)
                        })?;
                        let LinkPoint::Body { object, .. } =
                            self.sites.get(sname).copied().ok_or_else(|| {
                                format!("sensor '{name}' references unknown body site '{sname}'")
                            })?
                        else {
                            return Err(format!("wrench sensor '{name}' requires a body site"));
                        };
                        let link =
                            self.body_parent_joint
                                .get(&object)
                                .copied()
                                .ok_or_else(|| {
                                    format!("wrench sensor '{name}' site body has no parent joint")
                                })?;
                        Probe::Wrench { link }
                    }
                    "actuatorfrc" => {
                        let aname = child.attrs.get("actuator").ok_or_else(|| {
                            format!("sensor '{name}' <actuatorfrc> missing actuator")
                        })?;
                        let link = *self.actuators_by_name.get(aname).ok_or_else(|| {
                            format!("sensor '{name}' references unknown actuator '{aname}'")
                        })?;
                        Probe::Wrench { link }
                    }
                    _ => continue,
                };
                let pid = self.world.add_probe(probe);
                self.probes_by_name.insert(name, pid);
            }
        }
        Ok(())
    }
}

impl World {
    /// Parses a MuJoCo XML (MJCF) model description and spawns all bodies, geoms,
    /// joints, contact exclusions, spatial tendons, actuators, and sensors directly
    /// into `self` with origin at `root_pose`.
    pub fn load_mjcf(
        &mut self,
        xml: &str,
        default_material: MaterialId,
        root_pose: Pose,
    ) -> Result<RobotModel, String> {
        let root_node = parse_xml_doc(xml, "mujoco")?;
        let model_name = root_node
            .attrs
            .get("model")
            .cloned()
            .unwrap_or_else(|| "mjcf_model".into());

        let compiler = parse_compiler(&root_node)?;
        let defaults = collect_defaults(&root_node);

        for opt in root_node.children.iter().filter(|c| c.tag == "option") {
            if let Some(grav_str) = opt.attrs.get("gravity") {
                let g = parse_vec3(grav_str)?;
                self.gravity = root_pose.rotation.rotate(g);
            }
        }

        let worldbody = root_node
            .children
            .iter()
            .find(|c| c.tag == "worldbody")
            .ok_or("MJCF missing <worldbody> element")?;

        let mut builder = MjcfBuilder {
            world: self,
            default_material,
            compiler,
            defaults,
            links_by_name: HashMap::new(),
            body_objects: HashMap::new(),
            joints_by_name: HashMap::new(),
            joint_body: HashMap::new(),
            body_parent_joint: HashMap::new(),
            joint_specs: HashMap::new(),
            actuators_by_name: HashMap::new(),
            tendons_by_name: HashMap::new(),
            sites: HashMap::new(),
            site_normals: HashMap::new(),
            probes_by_name: HashMap::new(),
            geoms_by_name: HashMap::new(),
            material_cache: HashMap::new(),
            root_object: None,
            unnamed_counter: 0,
        };

        for child in &worldbody.children {
            match child.tag.as_str() {
                "geom" => {
                    let g = builder.parse_geom(child, None)?;
                    let gname = g
                        .name
                        .clone()
                        .unwrap_or_else(|| builder.next_name("world_geom"));
                    let g_world = compose_pose(root_pose, g.local_pose);
                    let id = builder
                        .world
                        .add_fixed(&gname, g.material, g.shape, g_world);
                    if let Some(ref raw_name) = g.name {
                        builder.geoms_by_name.insert(raw_name.clone(), id);
                    }
                    if builder.root_object.is_none() {
                        builder.root_object = Some(id);
                    }
                }
                "site" => {
                    builder.record_site(child, None, None, root_pose)?;
                }
                "body" => {
                    builder.spawn_body(child, None, &[], root_pose, None)?;
                }
                _ => {}
            }
        }

        builder.process_contacts(&root_node);
        builder.process_tendons(&root_node)?;
        builder.process_actuators(&root_node)?;
        builder.process_sensors(&root_node)?;

        let root = builder
            .root_object
            .ok_or("MJCF <worldbody> contains no bodies or geoms")?;

        Ok(RobotModel {
            name: model_name,
            root,
            links: builder.links_by_name,
            joints: builder.joints_by_name,
            actuators: builder.actuators_by_name,
            tendons: builder.tendons_by_name,
            sites: builder.sites,
            probes: builder.probes_by_name,
            geoms: builder.geoms_by_name,
        })
    }
}
