//! A jelly ball (E = 10 kPa) dropped 0.5 m: the share of its fall it climbs
//! back, by friction and by whether struck nodes get the impact law.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, TetMesh, World};
use phys_mechanics::{strain_energy, Mechanics, Settings};

fn main() {
    for (friction, impact_law) in [(0.5, true)] {
        let mut world = World::new();
        let floor = world.add_material(Material::new("floor", 2400.0, 30e9, 0.2, friction, 1.0));
        let jelly = world.add_material(Material::new("jelly", 1000.0, 1e4, 0.3, friction, 1.0));
        world.add_fixed(
            "floor",
            floor,
            Shape::cuboid(vec3(20.0, 1.0, 20.0)),
            Pose::at(vec3(0.0, -0.5, 0.0)),
        );
        let (r, drop) = (0.15, 0.5);
        let id = world.add_solid(
            "ball",
            jelly,
            TetMesh::ball(r, 0.03),
            Pose::at(vec3(0.0, r + drop, 0.0)),
        );
        let mut m = Mechanics::new(Settings {
            impact_law,
            ..Settings::default()
        });
        let pool = Pool::new(4);
        let start = world.object(id).centre_of_mass().unwrap().y;
        let e0 = world.totals();
        let (mut rising, mut peak, mut low) = (false, f64::MIN, f64::MAX);
        let mut _impacts = 0;
        for step in 0..120 {
            let rep = m.step(&pool, &mut world, 1.0 / 60.0);
            _impacts += rep.impacts;
            let o = world.object(id);
            let y = o.centre_of_mass().unwrap().y;
            let vy = o.momentum().y / o.mass();
            low = low.min(y);
            if vy > 0.05 {
                rising = true;
            }
            if rising {
                peak = peak.max(y);
            }
            let t = world.totals();
            if step % 3 == 0 && (15..45).contains(&step) {
                let w = strain_energy(&world);
                println!("  {step:>3} y {y:.4} vy {vy:+.3}  KE {:.2} (of it in the centre {:.2})  PE {:.2}  strain {w:.2}  total {:.2} (start {:.2})", t.kinetic_energy, 0.5 * o.mass() * (o.momentum() / o.mass()).length_squared(), t.gravitational_energy, t.kinetic_energy + t.gravitational_energy + w, e0.gravitational_energy);
            }
        }
        let climbed = (peak - low) / (start - low);
        println!("impact law {impact_law}: climbs back {climbed:.2} of its fall\n");
        let _ = Vec3::ZERO;
    }
}
