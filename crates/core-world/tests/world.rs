use core_math::constants::STANDARD_GRAVITY;
use core_math::{vec3, Pose, Quat, Vec3};
use core_space::Shape;
use core_world::{
    library, Column, Columns, JointDrive, LinkPoint, Material, Probe, Property, Representation,
    TetMesh, Thermal, World,
};

#[test]
fn meshes_are_conforming_and_closed() {
    for mesh in [
        TetMesh::cuboid(vec3(1.0, 0.2, 0.3), 0.1),
        TetMesh::ball(0.5, 0.1),
    ] {
        assert!(mesh.is_valid());
        let surface = mesh.surface();
        // A closed, consistently wound surface: every edge is shared by
        // exactly two faces, in opposite directions, and Euler's formula
        // holds for a sphere-like surface (V − E + F = 2).
        let mut directed = std::collections::HashMap::new();
        for f in &surface.faces {
            for k in 0..3 {
                *directed.entry((f[k], f[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        assert!(directed
            .iter()
            .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1)));
        let (v, e, f) = (
            surface.nodes.len() as i64,
            surface.edges.len() as i64,
            surface.faces.len() as i64,
        );
        assert_eq!(v - e + f, 2);
        // The divergence theorem: the outward surface encloses the volume.
        let enclosed: f64 = surface
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|i| mesh.rest[i as usize]);
                a.dot(b.cross(c)) / 6.0
            })
            .sum();
        assert!((enclosed - mesh.volume()).abs() < 1e-12 * mesh.volume().max(1.0));
    }
    let bar = TetMesh::cuboid(vec3(1.0, 0.2, 0.3), 0.1);
    assert!((bar.volume() - 0.06).abs() < 1e-14);
    assert_eq!(bar.tets.len(), 10 * 2 * 3 * 6);
}

#[test]
fn rigid_and_solid_agree_on_mass_and_momentum() {
    let mut world = World::new();
    let steel = world.add_material(library::structural_steel());
    let size = vec3(0.4, 0.2, 0.3);
    let pose = Pose::new(
        vec3(1.0, 2.0, -0.5),
        Quat::from_axis_angle(vec3(1.0, 1.0, 0.0).normalize(), 0.6),
    );
    let rigid = world.add_rigid("rigid", steel, Shape::cuboid(size), pose);
    let solid = world.add_solid("solid", steel, TetMesh::cuboid(size, 0.05), pose);
    let (v, w) = (vec3(0.3, -1.0, 2.0), vec3(0.0, 4.0, -1.0));
    world.set_velocity(rigid, v, w);
    world.set_velocity(solid, v, w);

    let (r, s) = (world.object(rigid), world.object(solid));
    let mass = 7850.0 * 0.4 * 0.2 * 0.3;
    assert!((r.mass() - mass).abs() < 1e-9 && (s.mass() - mass).abs() < 1e-9);
    assert!((s.centre_of_mass().unwrap() - pose.position).length() < 1e-12);
    assert!((r.momentum() - s.momentum()).length() < 1e-9);
    // A lumped-mass mesh under-represents the inertia slightly, converging
    // with refinement; within a few percent at this resolution.
    assert!((r.kinetic_energy() / s.kinetic_energy() - 1.0).abs() < 0.03);
    assert!(
        (r.angular_momentum() - s.angular_momentum()).length() / r.angular_momentum().length()
            < 0.03
    );
}

#[test]
fn totals_and_anchors() {
    let mut world = World::new();
    let rubber = world.add_material(library::rubber());
    world.add_fixed("ground", rubber, Shape::ground(0.0), Pose::IDENTITY);
    let bar = world.add_solid(
        "bar",
        rubber,
        TetMesh::cuboid(vec3(1.0, 0.1, 0.1), 0.05),
        Pose::at(vec3(0.0, 1.0, 0.0)),
    );
    let anchor = world.anchor(bar, |p| p.x < -0.49);
    assert!(anchor.is_some());
    assert!(world.anchor(bar, |p| p.x < -10.0).is_none());
    let core_world::Link::Anchor { nodes, .. } = &world.links()[0] else {
        panic!("expected anchor link");
    };
    assert_eq!(nodes.len(), 3 * 3);

    let totals = world.totals();
    let mass = 920.0 * 0.01;
    assert!((totals.mass - mass).abs() < 1e-12);
    assert!((totals.gravitational_energy - mass * STANDARD_GRAVITY * 1.0).abs() < 1e-9);
    assert_eq!(totals.kinetic_energy, 0.0);
}

#[test]
fn materials_and_properties() {
    let steel = library::structural_steel();
    let (mu, lambda) = steel.lame(293.15);
    assert!((mu - 200e9 / 2.6).abs() < 1.0);
    assert!((lambda - 200e9 * 0.3 / (1.3 * 0.4)).abs() < 1.0);
    assert!((steel.bar_wave_speed(293.15) - 5047.6).abs() < 1.0);

    // Softening with temperature, held beyond the table.
    let hot = Material {
        youngs: Property::Table(vec![(293.15, 200e9), (873.15, 100e9)]),
        ..steel.clone()
    };
    assert_eq!(hot.youngs.at(0.0), 200e9);
    assert!((hot.youngs.at(583.15) - 150e9).abs() < 1.0);
    assert_eq!(hot.youngs.at(2000.0), 100e9);
    assert!((steel.friction_with(&library::ice()) - (0.5f64 * 0.05).sqrt()).abs() < 1e-15);
}

#[test]
fn columns_appear_on_demand() {
    let mut columns = Columns::new(4);
    assert!(columns.get(Column::TEMPERATURE).is_none());
    columns.ensure(Column::TEMPERATURE, 293.15)[2] = 400.0;
    assert_eq!(
        columns.get(Column::TEMPERATURE).unwrap(),
        &[293.15, 293.15, 400.0, 293.15]
    );
    assert_eq!(columns.ensure(Column::TEMPERATURE, 0.0)[2], 400.0);
    assert_eq!(
        columns.names().collect::<Vec<_>>(),
        vec![Column::TEMPERATURE]
    );
}

#[test]
fn solids_start_at_rest_where_placed() {
    let mut world = World::new();
    let gel = world.add_material(library::gelatin());
    let pose = Pose::new(vec3(0.0, 3.0, 0.0), Quat::from_axis_angle(Vec3::Z, 0.3));
    let id = world.add_solid("ball", gel, TetMesh::ball(0.2, 0.05), pose);
    let Representation::Solid(s) = &world.object(id).representation else {
        unreachable!()
    };
    for (p, r) in s.nodes.position.iter().zip(&s.mesh.rest) {
        assert!((pose.inverse_transform_point(*p) - *r).length() < 1e-12);
    }
    assert!(s.nodes.velocity.iter().all(|v| *v == Vec3::ZERO));
}

#[test]
fn quadratic_meshes() {
    let linear = TetMesh::cuboid(vec3(1.0, 0.2, 0.3), 0.1);
    let mesh = linear.quadratic();
    assert!(mesh.is_valid() && mesh.is_quadratic());
    assert_eq!(mesh.tets.len(), linear.tets.len());
    // Every midpoint is halfway along its edge, and shared.
    for (t, m) in mesh.tets.iter().zip(&mesh.midpoints) {
        for (k, (i, j)) in core_world::EDGES.into_iter().enumerate() {
            let expected = (mesh.rest[t[i] as usize] + mesh.rest[t[j] as usize]) * 0.5;
            assert!((mesh.rest[m[k] as usize] - expected).length() < 1e-15);
        }
    }
    let unique_edges: std::collections::HashSet<(u32, u32)> = mesh
        .tets
        .iter()
        .flat_map(|t| core_world::EDGES.map(|(i, j)| (t[i].min(t[j]), t[i].max(t[j]))))
        .collect();
    assert_eq!(mesh.rest.len(), linear.rest.len() + unique_edges.len());
    // The surface is still closed and encloses the volume.
    let surface = mesh.surface();
    let enclosed: f64 = surface
        .faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|i| mesh.rest[i as usize]);
            a.dot(b.cross(c)) / 6.0
        })
        .sum();
    assert!((enclosed - mesh.volume()).abs() < 1e-12);
    // Mass shares add up, and the lumped mass is the body's.
    assert!((mesh.mass_shares().iter().sum::<f64>() - 1.0).abs() < 1e-15);
    let mut world = World::new();
    let gel = world.add_material(library::gelatin());
    let id = world.add_solid("bar", gel, mesh.clone(), Pose::IDENTITY);
    assert!((world.object(id).mass() - 1050.0 * 0.06).abs() < 1e-9);
}

#[test]
fn urdf_loader_tendon_probes_and_phase_change_properties() {
    let mut world = World::new();
    let steel = world.add_material(library::structural_steel().with_fracture(1e5, 5e4, 1e-3));
    assert!(world.material(steel).fracture.is_some());

    let wax = Thermal {
        specific_heat: 2000.0,
        ..Thermal::default()
    }
    .with_phase_change(330.0, 200_000.0, 10.0);
    // Integral of effective_specific_heat over [325, 335] equals c_p * 10 + latent_heat.
    let n = 1000;
    let d_t = 10.0 / n as f64;
    let total_enthalpy: f64 = (0..n)
        .map(|k| wax.effective_specific_heat(325.0 + (k as f64 + 0.5) * d_t) * d_t)
        .sum();
    assert!(
        (total_enthalpy - (2000.0 * 10.0 + 200_000.0)).abs() < 1.0,
        "enthalpy integral mismatch: {total_enthalpy}"
    );

    let urdf = r#"
    <robot name="two_link_arm">
      <link name="base_link">
        <inertial><mass value="2.0"/></inertial>
        <collision>
          <origin xyz="0 0.1 0" rpy="0 0 0"/>
          <geometry><box size="0.2 0.2 0.2"/></geometry>
        </collision>
      </link>
      <link name="upper_arm">
        <inertial>
          <mass value="1.5"/>
          <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
        </inertial>
        <collision>
          <origin xyz="0 0 0.2" rpy="0 0 0"/>
          <geometry><cylinder radius="0.04" length="0.4"/></geometry>
        </collision>
      </link>
      <joint name="shoulder" type="revolute">
        <parent link="base_link"/>
        <child link="upper_arm"/>
        <origin xyz="0 0.2 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.5" upper="1.5" effort="50.0"/>
        <dynamics damping="2.0"/>
      </joint>
    </robot>
    "#;
    let robot = world
        .load_urdf(urdf, steel, Pose::IDENTITY, true)
        .expect("valid URDF");
    assert_eq!(robot.name, "two_link_arm");
    let shoulder = robot.joint("shoulder").unwrap();
    let base_id = robot.link("base_link").unwrap();
    let arm_id = robot.link("upper_arm").unwrap();
    assert!(!world.object(base_id).is_dynamic());
    assert!((world.object(arm_id).mass() - 1.5).abs() < 1e-12);
    assert!(world.joint_coordinate(shoulder).unwrap().abs() < 1e-12);

    // Tendon and probes register cleanly.
    let tendon = world.add_tendon(
        vec![
            LinkPoint::Body {
                object: base_id,
                local: Vec3::ZERO,
            },
            LinkPoint::Body {
                object: arm_id,
                local: Vec3::ZERO,
            },
        ],
        500.0,
        5.0,
        true,
        JointDrive::pd(0.25, 400.0, 2.0),
    );
    assert!(world.joint_coordinate(tendon).unwrap() > 0.1);
    let _p1 = world.add_probe(Probe::JointState { link: shoulder });
    let _p2 = world.add_probe(Probe::Wrench { link: shoulder });
    let _p3 = world.add_probe(Probe::Tactile {
        object: arm_id,
        local_origin: Vec3::ZERO,
        local_normal: Vec3::Y,
        radius: 0.2,
    });
    let _p4 = world.add_probe(Probe::Imu {
        object: arm_id,
        local: Vec3::ZERO,
    });
    assert_eq!(world.probes().len(), 4);
}
