//! Named, step-indexed input and reusable rigid-body behaviours.
//!
//! Input is scene data: step `n` reads the last event at or before `n`.
//! There is no wall clock or playback cursor to lose during a checkpoint seek.
use fusion_core::World;
use fusion_math::Vec3;
use serde::{Deserialize, Serialize};

/// One change to a scalar action, held until its next event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputEvent {
    /// Zero-based solver step on which this value first applies.
    pub step: u64,
    /// Dimensionless input in [-1, 1].
    pub value: f32,
}

/// A named scalar action with optional editor keyboard bindings and a take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputAction {
    /// Script identifier, read as `input.name`.
    pub name: String,
    /// Key for -1; empty means unbound. Opposite keys cancel.
    #[serde(default)]
    pub negative: String,
    /// Key for +1; empty means unbound.
    #[serde(default)]
    pub positive: String,
    /// Strictly increasing solver-step changes. Before the first event, zero.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<InputEvent>,
}

impl InputAction {
    /// A keyboard axis with an empty take.
    pub fn axis(name: &str, negative: &str, positive: &str) -> Self {
        Self {
            name: name.into(),
            negative: negative.into(),
            positive: positive.into(),
            events: Vec::new(),
        }
    }

    /// Sample the take without mutating it, also when seeking backwards.
    pub fn value(&self, step: u64) -> f32 {
        let end = self.events.partition_point(|event| event.step <= step);
        end.checked_sub(1).map_or(0.0, |i| self.events[i].value)
    }

    /// Record a change. Re-recording an earlier step replaces this action's future.
    /// Returns false for invalid input or an unchanged take.
    pub fn record(&mut self, step: u64, value: f32) -> bool {
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return false;
        }
        let end = self.events.partition_point(|event| event.step < step);
        let previous = end.checked_sub(1).map_or(0.0, |i| self.events[i].value);
        if self.events.get(end).is_none() && previous == value {
            return false;
        }
        if self.events.len() == end + 1
            && self.events[end].step == step
            && self.events[end].value == value
        {
            return false;
        }
        self.events.truncate(end);
        if previous != value {
            self.events.push(InputEvent { step, value });
        }
        true
    }
}

/// One action's recorded events inside a stored take.
///
/// Keyed by action *name* rather than by position, because a take outlives the
/// list it was recorded against: renaming an action orphans its stream instead
/// of silently handing those events to whichever action moved into that slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TakeStream {
    /// The [`InputAction::name`] these events belong to.
    pub action: String,
    /// Strictly increasing solver-step changes, as in [`InputAction::events`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<InputEvent>,
}

/// One stored performance: what every action did, under a name.
///
/// The *live* take is still [`InputAction::events`], which is what the solver
/// samples and what recording writes. A `Take` is that data parked under a
/// name so the next recording appends rather than overwrites — the limitation
/// CONTROLS.md recorded, and the reason takes are a list at all.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Take {
    /// Shown in the editor and unique within a scene.
    pub name: String,
    /// One entry per action that recorded anything. Silent actions are omitted
    /// rather than stored empty, so a take costs what it contains.
    #[serde(default, rename = "stream", skip_serializing_if = "Vec::is_empty")]
    pub streams: Vec<TakeStream>,
}

impl Take {
    /// Park the live events of `actions` under `name`.
    pub fn capture(name: impl Into<String>, actions: &[InputAction]) -> Self {
        Self {
            name: name.into(),
            streams: actions
                .iter()
                .filter(|a| !a.events.is_empty())
                .map(|a| TakeStream {
                    action: a.name.clone(),
                    events: a.events.clone(),
                })
                .collect(),
        }
    }

    /// Load this take into `actions`, clearing any action it does not mention.
    ///
    /// Clearing rather than leaving the previous events in place is the whole
    /// point: selecting a take must produce that take's run, not a blend of it
    /// with whatever was loaded before.
    pub fn apply(&self, actions: &mut [InputAction]) {
        for action in actions {
            match self.streams.iter().find(|s| s.action == action.name) {
                Some(stream) => action.events.clone_from(&stream.events),
                None => action.events.clear(),
            }
        }
    }

    /// Whether this take records nothing at all.
    pub fn is_empty(&self) -> bool {
        self.streams.iter().all(|s| s.events.is_empty())
    }
}

/// `stem 1`, `stem 2`, … — the first no stored take is using.
///
/// Always numbered, so the name a recording gets is predictable rather than
/// depending on whether it happens to be the first.
pub fn unused_take_name(stem: &str, takes: &[Take]) -> String {
    (1..)
        .map(|n| format!("{stem} {n}"))
        .find(|name| !takes.iter().any(|t| &t.name == name))
        .expect("an unused suffix always exists")
}

/// Check stored takes: named, uniquely, over ordered and finite events.
///
/// A stream naming an action the scene does not have is *not* an error. It is
/// how a take survives an action being renamed or removed, and [`Take::apply`]
/// already treats an unmatched action as silent.
pub fn validate_takes(takes: &[Take]) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    for take in takes {
        if take.name.trim().is_empty() || !names.insert(&take.name) {
            return Err(format!(
                "take {:?} must have a unique, non-empty name",
                take.name
            ));
        }
        let mut streams = std::collections::BTreeSet::new();
        for stream in &take.streams {
            if !streams.insert(&stream.action) {
                return Err(format!(
                    "take {} records action {} more than once",
                    take.name, stream.action
                ));
            }
            if stream
                .events
                .iter()
                .any(|e| !e.value.is_finite() || !(-1.0..=1.0).contains(&e.value))
                || stream.events.windows(2).any(|w| w[0].step >= w[1].step)
            {
                return Err(format!(
                    "take {} requires finite values in [-1, 1] and increasing steps for {}",
                    take.name, stream.action
                ));
            }
        }
    }
    Ok(())
}

/// Check action identifiers, keys and ordered, finite takes before building.
pub fn validate_actions(actions: &[InputAction]) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    for action in actions {
        let mut chars = action.name.chars();
        if !chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            || !names.insert(&action.name)
        {
            return Err(format!(
                "input action {:?} must have a unique script identifier",
                action.name
            ));
        }
        for key in [&action.negative, &action.positive] {
            if !key.is_empty() && !KEYS.contains(&key.as_str()) {
                return Err(format!(
                    "input action {} has unsupported key {key:?}",
                    action.name
                ));
            }
        }
        if action
            .events
            .iter()
            .any(|e| !e.value.is_finite() || !(-1.0..=1.0).contains(&e.value))
            || action.events.windows(2).any(|w| w[0].step >= w[1].step)
        {
            return Err(format!(
                "input action {} requires finite values in [-1, 1] and increasing steps",
                action.name
            ));
        }
    }
    Ok(())
}

/// Portable keyboard binding names understood by the editor.
pub const KEYS: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "up",
    "down", "left", "right", "space",
];

/// Standard movement actions. World +Y is up, forward is -Z.
pub fn movement_actions() -> Vec<InputAction> {
    vec![
        InputAction::axis("move_x", "a", "d"),
        InputAction::axis("move_y", "q", "e"),
        InputAction::axis("move_z", "w", "s"),
    ]
}

/// A force-driven controller. It never teleports or overwrites velocity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForceController {
    /// Named scalar actions for world/local X, Y and Z.
    pub actions: [String; 3],
    /// Maximum force at full input, newtons.
    pub force: f32,
    /// Interpret axes in the object's rotating frame.
    #[serde(default)]
    pub local: bool,
}
impl Default for ForceController {
    fn default() -> Self {
        Self {
            actions: ["move_x".into(), "move_y".into(), "move_z".into()],
            force: 10.0,
            local: false,
        }
    }
}

/// A constant force and torque, optionally scaled by a named action.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppliedForce {
    /// Force at the centre of mass, newtons.
    pub force: [f32; 3],
    /// Torque about the centre of mass, newton metres.
    pub torque: [f32; 3],
    /// Empty applies continuously; otherwise multiplied by this scalar action.
    #[serde(default)]
    pub action: String,
    /// Rotate both vectors by the object's orientation.
    #[serde(default)]
    pub local: bool,
}

/// Reusable behaviours authored on one object and evaluated every solver step.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Components {
    /// Optional input-driven force controller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controller: Option<ForceController>,
    /// Optional force/torque source, independent of the controller and script.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<AppliedForce>,
    /// Expose this body's motion and mechanical energy as a probe.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub probe: bool,
}
impl Components {
    /// Whether the object has no additional behaviours.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
    /// Validate finite coefficients and references to existing input actions.
    pub fn validate(&self, actions: &[InputAction]) -> Result<(), String> {
        let exists = |name: &str| actions.iter().any(|a| a.name == name);
        if let Some(c) = &self.controller {
            if !c.force.is_finite() || c.force < 0.0 || c.actions.iter().any(|a| !exists(a)) {
                return Err(
                    "controller needs a finite nonnegative force and three existing actions".into(),
                );
            }
        }
        if let Some(f) = &self.force {
            if f.force.iter().chain(&f.torque).any(|v| !v.is_finite())
                || (!f.action.is_empty() && !exists(&f.action))
            {
                return Err("force needs finite vectors and an existing action (or none)".into());
            }
        }
        Ok(())
    }
}

/// Components resolved to a rigid-body index.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyComponents {
    /// Index in the built world's rigid bodies.
    pub body: usize,
    /// Authored settings, shared by interactive and headless runs.
    pub components: Components,
}

/// Read a scalar action. Missing actions return None, catching script typos.
pub fn value(actions: &[InputAction], name: &str, step: u64) -> Option<f32> {
    actions
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.value(step))
}

pub(crate) fn apply(
    world: &mut World,
    components: &[BodyComponents],
    actions: &[InputAction],
    step: u64,
    dt: f32,
) {
    for entry in components {
        let i = entry.body;
        if i >= world.bodies.len() || !world.bodies.is_dynamic(i) {
            continue;
        }
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        let rotation = world.bodies.orientation[i];
        if let Some(c) = &entry.components.controller {
            let v = Vec3::from_array(
                c.actions
                    .each_ref()
                    .map(|a| value(actions, a, step).unwrap_or(0.0)),
            );
            // Cap the vector magnitude: diagonals do not get extra thrust.
            let v = v / v.length().max(1.0) * c.force;
            force += if c.local { rotation * v } else { v };
        }
        if let Some(f) = &entry.components.force {
            let scale = if f.action.is_empty() {
                1.0
            } else {
                value(actions, &f.action, step).unwrap_or(0.0)
            };
            let fvec = Vec3::from_array(f.force) * scale;
            let tvec = Vec3::from_array(f.torque) * scale;
            force += if f.local { rotation * fvec } else { fvec };
            torque += if f.local { rotation * tvec } else { tvec };
        }
        if force != Vec3::ZERO || torque != Vec3::ZERO {
            world.bodies.wake(i);
            let centre = world.bodies.translation(i);
            world.bodies.apply_impulse_at(i, force * dt, centre);
            let delta = world.bodies.inv_inertia_world(i) * (torque * dt);
            world.bodies.angular_velocity[i] += delta;
        }
    }
}

/// Instantaneous rigid-body probe; energy uses the world's origin as height zero.
#[derive(Clone, Copy, Debug)]
pub struct ProbeReading {
    /// Solver step at which this state was sampled.
    pub step: u64,
    /// Time, seconds.
    pub time: f64,
    /// Body index.
    pub body: usize,
    /// Centre of mass, metres.
    pub position: Vec3,
    /// Linear velocity, metres per second.
    pub velocity: Vec3,
    /// Angular velocity, radians per second.
    pub spin: Vec3,
    /// Linear plus rotational kinetic energy, joules; zero for infinite mass.
    pub kinetic: f32,
    /// Gravitational potential energy, joules; zero for infinite mass.
    pub potential: f32,
}

impl crate::Simulation {
    /// Maximum number of retained probe readings, shared across all probes.
    pub const PROBE_CAPACITY: usize = 65_536;

    /// Read any rigid body's current state; no probe component is required.
    pub fn read_body(&self, body: usize) -> Option<ProbeReading> {
        let b = &self.world.bodies;
        if body >= b.len() {
            return None;
        }
        let position = b.translation(body);
        let velocity = b.linear_velocity[body];
        let spin = b.angular_velocity[body];
        let (kinetic, potential) = if b.is_dynamic(body) {
            (
                0.5 * b.mass(body) * velocity.length_squared()
                    + 0.5 * spin.dot(b.inertia_world(body) * spin),
                -b.mass(body) * self.world.gravity.dot(position),
            )
        } else {
            (0.0, 0.0)
        };
        Some(ProbeReading {
            step: self.step_index(),
            time: self.time(),
            body,
            position,
            velocity,
            spin,
            kinetic,
            potential,
        })
    }

    /// Recorded probe samples, one per enabled probe per solver step.
    /// Oldest readings expire at capacity. Reset/restore clears this output history;
    /// it does not influence simulation state or checkpoint hashes.
    pub fn probe_samples(&self) -> &std::collections::VecDeque<ProbeReading> {
        &self.probe_samples
    }

    pub(crate) fn sample_probes(&mut self) {
        for entry in &self.components {
            if !entry.components.probe {
                continue;
            }
            if let Some(reading) = self.read_body(entry.body) {
                if self.probe_samples.len() == Self::PROBE_CAPACITY {
                    self.probe_samples.pop_front();
                }
                self.probe_samples.push_back(reading);
            }
        }
    }
}
