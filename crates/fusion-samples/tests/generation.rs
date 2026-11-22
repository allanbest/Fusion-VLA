use fusion_core::{BodyDesc, World};
use fusion_math::vec3;
use fusion_project::{Project, SolverKind};
use fusion_samples::{generate, Config};
use std::path::PathBuf;
fn project(overlap: bool) -> Project {
    let mut w = World::new();
    w.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    w.add(BodyDesc::sphere(0.1).at(vec3(0.0, if overlap { 0.0 } else { 0.4 }, 0.0)));
    let mut p = Project::from_world(&w);
    p.solver.kind = SolverKind::Core;
    p
}
fn directory(name: &str) -> PathBuf {
    static ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "fusion-samples-{}-{name}-{}",
        std::process::id(),
        ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}
fn payload(path: PathBuf) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[10 + header..].to_vec()
}
#[test]
fn scheduling_does_not_change_samples_and_numpy_shape_is_complete() {
    let a = directory("one");
    let b = directory("two");
    let config = Config {
        trajectories: 4,
        steps: 12,
        seed: 14,
        ..Default::default()
    };
    let report = generate(&project(false), &a, &config).unwrap();
    assert_eq!(report.transitions, 48);
    generate(
        &project(false),
        &b,
        &Config {
            workers: 2,
            threads: 2,
            ..config
        },
    )
    .unwrap();
    let one = payload(a.join("states-000.npy"));
    let first = payload(b.join("states-000.npy"));
    let second = payload(b.join("states-001.npy"));
    let row = one.len() / 4;
    assert_eq!(first.len(), row * 2);
    assert_eq!(second.len(), row * 2);
    assert_eq!(&one[..row], &first[..row]);
    assert_eq!(&one[row..2 * row], &second[..row]);
    assert_eq!(&one[2 * row..3 * row], &first[row..]);
    assert_eq!(&one[3 * row..], &second[row..]);
    assert_ne!(&one[..row], &one[row..2 * row]);
    let manifest: toml::Value =
        toml::from_str(&std::fs::read_to_string(a.join("manifest.toml")).unwrap()).unwrap();
    assert_eq!(
        one.len(),
        4 * 13 * manifest["state_width"].as_integer().unwrap() as usize * 8
    );
    assert!(!a.join("INCOMPLETE").exists());
    assert!(!b.join("INCOMPLETE").exists());
    assert!(generate(&project(false), &a, &Config::default()).is_err());
    std::fs::remove_dir_all(a).unwrap();
    std::fs::remove_dir_all(b).unwrap();
}
#[test]
fn failure_is_explicit_and_cannot_produce_a_completed_dataset() {
    let dir = directory("failure");
    let result = generate(
        &project(true),
        &dir,
        &Config {
            trajectories: 1,
            steps: 3,
            velocity: 0.0,
            spin: 0.0,
            ..Default::default()
        },
    );
    assert!(result.unwrap_err().to_string().contains("solve rejected"));
    assert!(dir.join("INCOMPLETE").exists());
    assert!(!dir.join("manifest.toml").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn elastic_geometry_and_parallel_solver_produce_identical_shards() {
    use fusion_core::{SolidDesc, SolidMaterial, SolidParams};
    use fusion_math::Vec3;
    let mut w = World::new();
    w.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.0, 2.0, 0.0))
            .with_params(SolidParams::at(0.025).of(SolidMaterial::JELLY)),
    );
    let mut p = Project::from_world(&w);
    p.solver.kind = SolverKind::Core;
    let a = directory("fem-serial");
    let b = directory("fem-parallel");
    let c = Config {
        trajectories: 2,
        steps: 3,
        seed: 5,
        ..Default::default()
    };
    generate(&p, &a, &c).unwrap();
    generate(
        &p,
        &b,
        &Config {
            workers: 2,
            threads: 2,
            ..c
        },
    )
    .unwrap();
    let serial = payload(a.join("states-000.npy"));
    let first = payload(b.join("states-000.npy"));
    let second = payload(b.join("states-001.npy"));
    assert_eq!(&serial[..serial.len() / 2], first);
    assert_eq!(&serial[serial.len() / 2..], second);
    for part in ["rest", "mass", "tets", "midpoints"] {
        assert_eq!(
            payload(a.join(format!("object-0-{part}.npy"))),
            payload(b.join(format!("object-0-{part}.npy")))
        );
    }
    std::fs::remove_dir_all(a).unwrap();
    std::fs::remove_dir_all(b).unwrap();
}
