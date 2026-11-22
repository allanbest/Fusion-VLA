//! Behaviour scripts, attached to bodies and run inside the step loop.
//!
//! [`fusion_script`] is the language; this is where it meets a rigid body.
//! Two things are decided here, and both of them matter more than the wiring:
//!
//! ## Scripts run in the simulation, not in the editor
//!
//! A script is part of the scene, so it has to run wherever the scene runs —
//! the editor, a demo binary, a Python driver, a headless research sweep. If
//! it ran in the GUI instead, a `.fusion` file would behave differently
//! depending on which program opened it, and that is precisely the two
//! front-ends drifting that PLAN.md §21 exists to prevent.
//!
//! ## A script asks or it tells, and which one depends on the body
//!
//! Writing `self.force` adds to an accumulator the solver then applies, like
//! gravity: the body is still solved, still collides, still conserves. Writing
//! `self.position` sets the pose outright, and that is only coherent on a
//! [`BodyType::Kinematic`] body — one the solver does not integrate. On a
//! dynamic body the solver would be deciding where it is at the same time as
//! the script, and the two would fight every step. So a pose write to a
//! dynamic body is refused with a message that says what to change, rather
//! than silently doing something that looks like a physics bug.
//!
//! ## Errors switch a script off
//!
//! A script that raises is disabled and its error kept. The alternative is an
//! error sixty times a second, which fills a console, costs a parse of the
//! message every step, and buries the first occurrence — the one with the
//! context. Fixing the script clears it, because recompiling makes a new
//! [`BodyScript`].

use std::collections::HashMap;

use fusion_core::{BodyType, World};
use fusion_math::{Quat, Vec3};
use fusion_script::{Host, Program, ScriptError, Value};

/// A script, the body it drives, and whatever it remembered.
#[derive(Clone, Debug)]
pub struct BodyScript {
    /// Index into the world's bodies.
    pub body: usize,
    /// The compiled program.
    pub program: Program,
    /// Variables the script assigned, carried between steps.
    pub locals: HashMap<String, Value>,
    /// Set once the script has failed. While this is set it does not run.
    pub failed: Option<ScriptError>,
}

impl BodyScript {
    /// Attach `program` to a body, with nothing remembered yet.
    pub fn new(body: usize, program: Program) -> Self {
        Self {
            body,
            program,
            locals: HashMap::new(),
            failed: None,
        }
    }
}

/// What one step of one script is allowed to touch.
struct BodyHost<'a> {
    world: &'a mut World,
    inputs: &'a [crate::control::InputAction],
    body: usize,
    time: f64,
    dt: f32,
    step: u64,
    /// Force asked for this step, applied after the script returns.
    force: Vec3,
    torque: Vec3,
    /// Set where the script wrote a pose, so the caller knows to derive a
    /// velocity from it.
    moved: bool,
}

impl Host for BodyHost<'_> {
    fn is_object(&self, name: &str) -> bool {
        name == "self" || name == "world" || name == "input"
    }

    fn get(&self, object: &str, field: &str) -> Option<Value> {
        if object == "input" {
            return crate::control::value(self.inputs, field, self.step).map(Value::Number);
        }
        let bodies = &self.world.bodies;
        let i = self.body;
        match (object, field) {
            ("world", "gravity") => Some(Value::Vector(self.world.gravity)),
            ("self", "position") => Some(Value::Vector(bodies.translation(i))),
            ("self", "rotation") => Some(Value::Vector(bodies.orientation[i].to_euler())),
            ("self", "velocity") => Some(Value::Vector(bodies.linear_velocity[i])),
            ("self", "spin") => Some(Value::Vector(bodies.angular_velocity[i])),
            ("self", "force") => Some(Value::Vector(self.force)),
            ("self", "torque") => Some(Value::Vector(self.torque)),
            ("self", "mass") => Some(Value::Number(bodies.mass(i))),
            ("self", "index") => Some(Value::Number(i as f32)),
            // Convenience, because "am I resting" is the single most common
            // thing a behaviour wants to know and deriving it from velocity in
            // script is four lines of arithmetic every time.
            ("self", "asleep") => Some(Value::Bool(bodies.sleeping[i])),
            _ => None,
        }
    }

    fn set(&mut self, object: &str, field: &str, value: Value) -> Result<(), String> {
        if object == "world" || object == "input" {
            // A body's script setting the world's gravity would make the scene
            // depend on which body happened to run last. Gravity is a property
            // of the world and is edited as one.
            return Err(format!("{object}.{field} is read-only from a script"));
        }
        let i = self.body;
        // Finite, or refused with the script's line. Script numbers are `f32`,
        // so `0/0` is NaN and `1e39` is already infinite by the time it gets
        // here; written into a body, either made it vanish with no error to
        // say why. The engine contains it — a NaN body does not spread to the
        // ones it touches — but a silent disappearance is the one outcome a
        // script author cannot debug.
        let vector = |value: Value| -> Result<Vec3, String> {
            let v = value
                .as_vector()
                .ok_or_else(|| format!("self.{field} is a vector"))?;
            if !(v.x.is_finite() && v.y.is_finite() && v.z.is_finite()) {
                return Err(format!(
                    "self.{field} must be finite, and this is ({}, {}, {})",
                    v.x, v.y, v.z
                ));
            }
            Ok(v)
        };

        match field {
            "force" => self.force = vector(value)?,
            "torque" => self.torque = vector(value)?,
            "position" => {
                self.require_kinematic("position")?;
                self.world.bodies.set_translation(i, vector(value)?);
                self.moved = true;
            }
            "rotation" => {
                self.require_kinematic("rotation")?;
                self.world.bodies.orientation[i] = Quat::from_euler(vector(value)?);
                self.moved = true;
            }
            "velocity" => {
                self.world.bodies.linear_velocity[i] = vector(value)?;
                // A body told to move is a body that is no longer settled.
                self.world.bodies.wake(i);
            }
            "spin" => {
                self.world.bodies.angular_velocity[i] = vector(value)?;
                self.world.bodies.wake(i);
            }
            "mass" | "index" | "asleep" => {
                return Err(format!("self.{field} is read-only"));
            }
            _ => return Err(format!("a body has no {field:?}")),
        }
        Ok(())
    }

    fn global(&self, name: &str) -> Option<Value> {
        match name {
            "t" => Some(Value::Number(self.time as f32)),
            "dt" => Some(Value::Number(self.dt)),
            "step" => Some(Value::Number(self.step as f32)),
            _ => None,
        }
    }
}

impl BodyHost<'_> {
    /// Refuse a pose write on a body the solver is also carrying.
    fn require_kinematic(&self, field: &str) -> Result<(), String> {
        if self.world.bodies.body_type[self.body] == BodyType::Dynamic {
            return Err(format!(
                "setting self.{field} on a dynamic body would fight the solver \u{2014} \
                 make it kinematic, or write self.force instead"
            ));
        }
        Ok(())
    }
}

/// Run every script once, and apply what they asked for.
///
/// Called at the top of a step, before collision detection, so that a pose a
/// script writes is the pose the contacts are found against. Running them
/// after the solve would find contacts against where the body was, and a
/// paddle would pass through what it was supposed to push.
pub fn run(
    scripts: &mut [BodyScript],
    world: &mut World,
    time: f64,
    dt: f32,
    step: u64,
) -> Vec<(usize, ScriptError)> {
    run_with_inputs(scripts, world, time, dt, step, &[])
}

/// Run scripts against the action values for this exact solver step.
pub fn run_with_inputs(
    scripts: &mut [BodyScript],
    world: &mut World,
    time: f64,
    dt: f32,
    step: u64,
    inputs: &[crate::control::InputAction],
) -> Vec<(usize, ScriptError)> {
    let mut errors = Vec::new();

    for (index, script) in scripts.iter_mut().enumerate() {
        if script.failed.is_some() || script.body >= world.bodies.len() {
            continue;
        }
        let before = world.bodies.pose(script.body);
        let mut host = BodyHost {
            world,
            inputs,
            body: script.body,
            time,
            dt,
            step,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            moved: false,
        };

        if let Err(error) = script.program.run(&mut host, &mut script.locals) {
            script.failed = Some(error.clone());
            errors.push((index, error));
            continue;
        }
        let (force, torque, moved) = (host.force, host.torque, host.moved);
        let body = script.body;

        // A force over a step is an impulse. The same explicit treatment
        // gravity gets, which is what keeps a scripted thruster and a
        // gravitational field behaving like the same kind of thing.
        //
        // Only a dynamic body has a mass for a force to divide by. A kinematic
        // one that asks for force is asking the solver for something it is not
        // doing on that body's behalf; ignoring it beats an error every step,
        // because the pose the script also sets is what actually moves it.
        let wanted = force != Vec3::ZERO || torque != Vec3::ZERO;
        if wanted && world.bodies.is_dynamic(body) {
            world.bodies.wake(body);
            let centre = world.bodies.translation(body);
            world.bodies.apply_impulse_at(body, force * dt, centre);
            if torque != Vec3::ZERO {
                let delta = world.bodies.inv_inertia_world(body) * (torque * dt);
                world.bodies.angular_velocity[body] += delta;
            }
        }

        // A kinematic body's velocity is whatever its motion implies. Deriving
        // it rather than asking the script for it is what makes a rising
        // platform carry a crate: the contact solve reads these arrays, and a
        // platform that reported zero velocity would slide out from under
        // everything resting on it.
        if moved && world.bodies.is_kinematic(body) && dt > 0.0 {
            let after = world.bodies.pose(body);
            world.bodies.linear_velocity[body] = (after.translation - before.translation) / dt;
            world.bodies.angular_velocity[body] =
                angular_velocity_between(before.rotation, after.rotation, dt);
        }
    }
    errors
}

/// The angular velocity that carries `from` to `to` in `dt`.
///
/// Taken from the relative rotation's axis and angle rather than by
/// differencing components, which is the version that stays correct through a
/// half turn and does not care which of `q` and `-q` either end happens to be.
fn angular_velocity_between(from: Quat, to: Quat, dt: f32) -> Vec3 {
    let mut delta = to * from.conjugate();
    // A quaternion and its negation are the same rotation; picking the one
    // with a positive scalar part is what makes the extracted angle the short
    // way round instead of the long way.
    if delta.w < 0.0 {
        delta = Quat::from_xyzw(-delta.x, -delta.y, -delta.z, -delta.w);
    }
    let vector = delta.vector_part();
    let sin_half = vector.length();
    if sin_half < 1e-7 {
        return Vec3::ZERO;
    }
    let angle = 2.0 * sin_half.atan2(delta.w);
    vector * (angle / (sin_half * dt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusion_core::BodyDesc;
    use fusion_math::vec3;

    fn world_with(desc: BodyDesc) -> World {
        let mut world = World::new();
        world.add(desc);
        world
    }

    /// A kinematic body's velocity comes from how far it actually moved.
    #[test]
    fn a_driven_body_reports_the_velocity_its_motion_implies() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE).driven());
        let mut scripts = vec![BodyScript::new(
            0,
            Program::compile("self.position.y = t * 2").expect("compiles"),
        )];
        let dt = 1.0 / 60.0;
        // Two steps: the first establishes a pose, the second a motion.
        run(&mut scripts, &mut world, dt as f64, dt, 1);
        run(&mut scripts, &mut world, 2.0 * dt as f64, dt, 2);

        let v = world.bodies.linear_velocity[0];
        assert!(
            (v.y - 2.0).abs() < 1e-3,
            "moved at {v:?}, expected 2 m/s up"
        );
        assert!(v.x.abs() < 1e-6 && v.z.abs() < 1e-6, "{v:?}");
    }

    /// A rotating paddle reports a spin, not nothing.
    #[test]
    fn a_turning_body_reports_a_spin() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE).driven());
        let mut scripts = vec![BodyScript::new(
            0,
            Program::compile("self.rotation.y = t").expect("compiles"),
        )];
        let dt = 1.0 / 60.0;
        run(&mut scripts, &mut world, dt as f64, dt, 1);
        run(&mut scripts, &mut world, 2.0 * dt as f64, dt, 2);

        let w = world.bodies.angular_velocity[0];
        assert!((w.y - 1.0).abs() < 1e-3, "spun at {w:?}, expected 1 rad/s");
    }

    /// Setting a pose on a dynamic body is refused, and the message says what
    /// to do instead.
    #[test]
    fn a_dynamic_body_cannot_be_teleported() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE));
        let mut scripts = vec![BodyScript::new(
            0,
            Program::compile("self.position.y = 5").expect("compiles"),
        )];
        let errors = run(&mut scripts, &mut world, 0.0, 1.0 / 60.0, 1);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].1.message.contains("kinematic"), "{}", errors[0].1);
        assert!(scripts[0].failed.is_some(), "a failed script keeps running");
    }

    /// A force reaches the body as an impulse over the step.
    #[test]
    fn a_force_moves_a_dynamic_body() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE).with_mass(2.0));
        let mut scripts = vec![BodyScript::new(
            0,
            Program::compile("self.force.y = 20").expect("compiles"),
        )];
        let dt = 0.5;
        run(&mut scripts, &mut world, 0.0, dt, 1);
        // impulse = F dt = 10 N s, over 2 kg, is 5 m/s.
        let v = world.bodies.linear_velocity[0];
        assert!((v.y - 5.0).abs() < 1e-4, "{v:?}");
    }

    /// A script that fails once does not run again, so a console does not
    /// fill with sixty copies of the same line a second.
    #[test]
    fn a_failed_script_stops() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE));
        let mut scripts = vec![BodyScript::new(
            0,
            Program::compile("x = nonsense").expect("compiles"),
        )];
        assert_eq!(run(&mut scripts, &mut world, 0.0, 0.01, 1).len(), 1);
        assert_eq!(run(&mut scripts, &mut world, 0.0, 0.01, 2).len(), 0);
    }

    /// A script reads the world it is in.
    #[test]
    fn a_script_can_see_the_scene_around_it() {
        let mut world = world_with(BodyDesc::cuboid(Vec3::ONE));
        world.gravity = vec3(0.0, -1.62, 0.0);
        let mut scripts = vec![BodyScript::new(
            0,
            // Cancel gravity exactly: the classic hover script.
            Program::compile("self.force = world.gravity * -self.mass").expect("compiles"),
        )];
        run(&mut scripts, &mut world, 0.0, 1.0, 1);
        let v = world.bodies.linear_velocity[0];
        assert!((v.y - 1.62).abs() < 1e-3, "{v:?}");
    }
}
