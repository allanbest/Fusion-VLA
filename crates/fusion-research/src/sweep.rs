//! Parameter sweeps: the same scene, run many times, one thing changed.
//!
//! §23's first item, and the one that turns the engine into an instrument.
//! "Does the tower fall over?" is a question about one run. "At what substep
//! count does the tower stop falling over?" is a question about twenty, and
//! answering it by editing a number and re-running by hand is how people end
//! up with results they cannot reproduce a week later.
//!
//! ## What a sweep is, concretely
//!
//! A base [`Project`] — scene data, exactly what a `.fusion` file holds — plus
//! axes. An axis is a name, a list of values, and a way to apply one value to
//! a project. The sweep is the cartesian product, run in order, each producing
//! a [`RunRecord`].
//!
//! The base being a `Project` is what makes a sweep reproducible: every run in
//! it is a scene that could have been saved to disk and re-run on its own, and
//! [`Sweep::project_for`] hands you exactly that. A sweep that mutated a live
//! `Simulation` instead would produce numbers nobody could get back to.

use std::collections::BTreeMap;

use fusion_project::Project;
use fusion_sim::Simulation;
use serde::{Deserialize, Serialize};

use crate::metrics::{ConservationSummary, ConservationTrace};

/// How a sweep puts one parameter value into a scene.
pub type ApplyValue = Box<dyn Fn(&mut Project, f64)>;

/// One axis of a sweep: a named parameter and the values to try.
pub struct Axis {
    /// What to call it in the results.
    pub name: String,
    /// The values, in the order they will be run.
    pub values: Vec<f64>,
    /// How to put one value into a project.
    ///
    /// A closure rather than a path into the struct, because the interesting
    /// parameters are not always fields — "drop height" is a position, "how
    /// many boxes" is the length of a list — and a sweep that could only vary
    /// fields would miss most of the questions worth asking.
    pub apply: ApplyValue,
}

impl Axis {
    /// An axis over a named parameter.
    pub fn new(
        name: impl Into<String>,
        values: impl IntoIterator<Item = f64>,
        apply: impl Fn(&mut Project, f64) + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            values: values.into_iter().collect(),
            apply: Box::new(apply),
        }
    }
}

/// What one run of a sweep produced.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    /// The parameter values this run used, by axis name.
    pub parameters: BTreeMap<String, f64>,
    /// Steps taken.
    pub steps: u64,
    /// Simulated seconds covered.
    pub simulated_time: f64,
    /// Wall-clock milliseconds the run took. Reported separately from
    /// simulated time, never mixed with it (§19).
    pub wall_ms: f64,
    /// State hash at the end, which is what makes a row re-checkable: run the
    /// same scene again and this number has to come back.
    pub state_hash: u64,
    /// Whether the run stayed finite.
    pub stable: bool,
    /// Conservation over the run.
    pub conservation: Option<ConservationSummary>,
    /// Whatever the caller measured, by name. The answer to the question the
    /// sweep was asked.
    pub measurements: BTreeMap<String, f64>,
}

/// The results of a whole sweep.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SweepResults {
    /// Axis names, in the order they vary.
    pub axes: Vec<String>,
    /// One record per combination.
    pub runs: Vec<RunRecord>,
}

impl SweepResults {
    /// The run that scored highest on a measurement.
    pub fn best_by(&self, measurement: &str) -> Option<&RunRecord> {
        self.runs
            .iter()
            .filter(|r| r.measurements.contains_key(measurement))
            .max_by(|a, b| a.measurements[measurement].total_cmp(&b.measurements[measurement]))
    }

    /// The run that scored lowest.
    pub fn worst_by(&self, measurement: &str) -> Option<&RunRecord> {
        self.runs
            .iter()
            .filter(|r| r.measurements.contains_key(measurement))
            .min_by(|a, b| a.measurements[measurement].total_cmp(&b.measurements[measurement]))
    }

    /// As a table, one row per run: the form a person reads and a spreadsheet
    /// imports.
    pub fn to_csv(&self) -> String {
        let mut columns: Vec<String> = self.axes.clone();
        let mut measurements: Vec<String> = Vec::new();
        for run in &self.runs {
            for key in run.measurements.keys() {
                if !measurements.contains(key) {
                    measurements.push(key.clone());
                }
            }
        }
        columns.extend(measurements.iter().cloned());
        let mut out = format!(
            "{},steps,sim_s,wall_ms,state_hash,stable,energy_delta,closed,momentum_drift\n",
            columns.join(",")
        );
        for run in &self.runs {
            for axis in &self.axes {
                out.push_str(&format!(
                    "{},",
                    run.parameters.get(axis).copied().unwrap_or(f64::NAN)
                ));
            }
            for key in &measurements {
                out.push_str(&format!(
                    "{},",
                    run.measurements.get(key).copied().unwrap_or(f64::NAN)
                ));
            }
            let conservation = run.conservation;
            let energy = conservation.map(|c| c.energy_delta()).unwrap_or(f64::NAN);
            let closed = conservation.map(|c| c.closed).unwrap_or(false);
            // Left blank where the scene is not closed, rather than filled
            // with a number that is not a measurement of anything. A momentum
            // "drift" for a body resting on a floor is the floor holding it
            // up, and a column of those in a spreadsheet is how a tool teaches
            // somebody a wrong thing.
            let momentum = match conservation {
                Some(c) if c.closed => format!("{:.9}", c.momentum_drift),
                _ => String::new(),
            };
            out.push_str(&format!(
                "{},{:.6},{:.3},{:#018x},{},{:.6},{},{}\n",
                run.steps,
                run.simulated_time,
                run.wall_ms,
                run.state_hash,
                run.stable,
                energy,
                closed,
                momentum
            ));
        }
        out
    }
}

/// A sweep waiting to be run.
pub struct Sweep {
    base: Project,
    axes: Vec<Axis>,
    steps: u64,
    sample_every: u64,
}

impl Sweep {
    /// A sweep over `base`, running each combination for `steps` steps.
    pub fn new(base: Project, steps: u64) -> Self {
        Self {
            base,
            axes: Vec::new(),
            steps,
            sample_every: 10,
        }
    }

    /// Add an axis. Combinations are the cartesian product of every axis.
    pub fn axis(mut self, axis: Axis) -> Self {
        self.axes.push(axis);
        self
    }

    /// Record conservation every `n` steps. Ten by default: often enough to
    /// find the step something went wrong on, rare enough that a long run does
    /// not spend its time summing energies.
    pub fn sample_every(mut self, n: u64) -> Self {
        self.sample_every = n.max(1);
        self
    }

    /// How many runs this sweep will perform.
    pub fn len(&self) -> usize {
        self.axes.iter().map(|a| a.values.len().max(1)).product()
    }

    /// True where there is nothing to run.
    pub fn is_empty(&self) -> bool {
        self.axes.is_empty() || self.axes.iter().any(|a| a.values.is_empty())
    }

    /// The project for one combination, by index.
    ///
    /// Public because a sweep result should always be reachable as a scene:
    /// this is what you save when one row of the table is the interesting one.
    pub fn project_for(&self, index: usize) -> Project {
        let mut project = self.base.clone();
        let mut remainder = index;
        for axis in &self.axes {
            let count = axis.values.len().max(1);
            let value = axis.values[remainder % count];
            remainder /= count;
            (axis.apply)(&mut project, value);
        }
        project
    }

    /// The parameter values for one combination.
    fn parameters_for(&self, index: usize) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        let mut remainder = index;
        for axis in &self.axes {
            let count = axis.values.len().max(1);
            out.insert(axis.name.clone(), axis.values[remainder % count]);
            remainder /= count;
        }
        out
    }

    /// Run every combination, measuring each with `measure`.
    ///
    /// `measure` is handed the finished simulation and returns whatever the
    /// question was — settled height, particle count, worst penetration. It
    /// runs after the steps, so it sees the end state; anything that has to be
    /// watched during a run belongs in the conservation trace instead.
    pub fn run(
        &self,
        mut measure: impl FnMut(&Simulation) -> BTreeMap<String, f64>,
    ) -> Result<SweepResults, fusion_project::ProjectError> {
        let mut results = SweepResults {
            axes: self.axes.iter().map(|a| a.name.clone()).collect(),
            runs: Vec::with_capacity(self.len()),
        };

        for index in 0..self.len() {
            let project = self.project_for(index);
            let mut sim = project.build()?;
            let mut trace = ConservationTrace::new();
            trace.observe(&sim);

            let started = std::time::Instant::now();
            for step in 0..self.steps {
                sim.step();
                if (step + 1) % self.sample_every == 0 {
                    trace.observe(&sim);
                }
            }
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            trace.observe(&sim);

            results.runs.push(RunRecord {
                parameters: self.parameters_for(index),
                steps: self.steps,
                simulated_time: sim.time(),
                wall_ms,
                state_hash: fusion_sim::state_hash(&sim),
                stable: sim.is_stable(),
                conservation: trace.summary(),
                measurements: measure(&sim),
            });
        }
        Ok(results)
    }
}
