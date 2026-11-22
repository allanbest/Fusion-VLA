//! Comparing two runs: where they diverge, and when.
//!
//! Two tools that look similar and answer opposite questions.
//!
//! [`determinism_diff`] runs *the same* scene twice and expects to find
//! nothing. §23: "run twice, hash state per step, report the first divergent
//! step and body. Turns the worst bug class in parallel physics into a CI
//! test." The engine already promises this (`Determinism::GUARANTEE`) and
//! already has tests asserting two runs agree — what it did not have is the
//! thing that says *where* they stopped agreeing, which is the difference
//! between knowing you have a bug and being able to find it.
//!
//! [`compare`] runs two *different* scenes — one parameter changed — and
//! expects to find something. §23: "two runs, one parameter changed, plot
//! where and when they diverge". A physics change is never a compile error and
//! rarely a crash; it is a number that starts drifting on step four hundred,
//! and this is what makes that visible.
//!
//! ## Why a hash and a distance
//!
//! The hash finds the first step where two runs stop being bit-identical,
//! which is the right question for determinism and the wrong one for
//! everything else — two physically identical runs can differ in the last bit
//! of one velocity. The distance finds the first step where they stop being
//! *physically* the same, which needs a tolerance and therefore a judgement.
//! Both are reported, because "identical until step 400, then 3 mm apart by
//! step 900" is a description and either number alone is not.

use fusion_sim::{state_hash, Simulation};
use serde::{Deserialize, Serialize};

/// What running one scene twice found.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeterminismReport {
    /// Steps both runs were advanced.
    pub steps: u64,
    /// First step whose state hash differed, if any.
    pub first_divergent_step: Option<u64>,
    /// Which body differed most at that step, and by how far.
    pub first_divergent_body: Option<usize>,
    /// How far apart that body was, metres.
    pub separation: f32,
}

impl DeterminismReport {
    /// True where the two runs were bit-identical the whole way.
    pub fn reproduced(&self) -> bool {
        self.first_divergent_step.is_none()
    }
}

/// Run the same scene twice and report the first step where the two disagree.
///
/// `build` is called once per run and must produce the same simulation both
/// times; anything it captures is part of the experiment.
///
/// A failure here is not a tolerance to widen. Fusion guarantees bit-identical
/// replay for the same build on the same machine (§34), so a divergent step is
/// a defect — usually a container iterated in a non-deterministic order, or a
/// sum whose order depends on something the scene does not record.
pub fn determinism_diff(mut build: impl FnMut() -> Simulation, steps: u64) -> DeterminismReport {
    let mut a = build();
    let mut b = build();
    let mut report = DeterminismReport {
        steps,
        first_divergent_step: None,
        first_divergent_body: None,
        separation: 0.0,
    };

    for step in 0..steps {
        a.step();
        b.step();
        if state_hash(&a) == state_hash(&b) {
            continue;
        }
        // Found the step. Now find the body, which is what makes the report
        // actionable: "step 412" is a place to put a breakpoint, "step 412,
        // body 37" is a place to look.
        let (mut worst, mut which) = (0.0f32, None);
        for i in 0..a.world.len().min(b.world.len()) {
            let d = (a.world.bodies.translation(i) - b.world.bodies.translation(i)).length();
            if d >= worst {
                worst = d;
                which = Some(i);
            }
        }
        report.first_divergent_step = Some(step + 1);
        report.first_divergent_body = which;
        report.separation = worst;
        break;
    }
    report
}

/// What comparing two different runs found.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Steps both runs were advanced.
    pub steps: u64,
    /// First step at which the two stopped being bit-identical.
    pub first_bitwise_difference: Option<u64>,
    /// First step at which some body moved further than the tolerance from
    /// its counterpart.
    pub first_visible_difference: Option<u64>,
    /// The tolerance that "visible" was measured against, metres.
    pub tolerance: f32,
    /// Largest separation reached, and where.
    pub max_separation: f32,
    /// The step it happened on.
    pub max_separation_step: u64,
    /// The body it happened to.
    pub max_separation_body: Option<usize>,
    /// Separation over time, one entry per step: the plot §23 asks for, as
    /// numbers rather than as a picture.
    pub separation: Vec<f32>,
}

/// Run two scenes side by side and report where they part company.
///
/// The scenes are stepped in lockstep, so `separation[i]` is the distance
/// after step `i + 1` and the whole vector is the divergence curve.
pub fn compare(
    a: &mut Simulation,
    b: &mut Simulation,
    steps: u64,
    tolerance: f32,
) -> ComparisonReport {
    let mut report = ComparisonReport {
        steps,
        tolerance,
        ..Default::default()
    };

    for step in 0..steps {
        a.step();
        b.step();

        if report.first_bitwise_difference.is_none() && state_hash(a) != state_hash(b) {
            report.first_bitwise_difference = Some(step + 1);
        }

        let (mut worst, mut which) = (0.0f32, None);
        for i in 0..a.world.len().min(b.world.len()) {
            let d = (a.world.bodies.translation(i) - b.world.bodies.translation(i)).length();
            if d >= worst {
                worst = d;
                which = Some(i);
            }
        }
        report.separation.push(worst);
        if worst > report.max_separation {
            report.max_separation = worst;
            report.max_separation_step = step + 1;
            report.max_separation_body = which;
        }
        if report.first_visible_difference.is_none() && worst > tolerance {
            report.first_visible_difference = Some(step + 1);
        }
    }
    report
}
