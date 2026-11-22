//! Fusion L2 — behaviour scripts on objects, in a deterministic subset of
//! Python.
//!
//! ```text
//! # a body that bobs on a wave
//! self.position.y = 2 + sin(t * 3) * 0.5
//!
//! # a body that is pushed rather than placed
//! self.force.y = 40 * max(0, sin(t * 2))
//! ```
//!
//! ## What this is, and what it is not
//!
//! It is a real language with a real tokenizer, parser and evaluator:
//! expressions with Python's precedence, `if`/`elif`/`else`, `while`,
//! variables that persist between steps, significant indentation, and error
//! messages that name a line. It is **not** CPython, and it never runs
//! arbitrary Python. There are no imports, no classes, no functions, no
//! strings, no collections, and no way to reach the machine it is running on.
//!
//! That is a deliberate trade and it is worth stating plainly, because
//! "embedded Python" usually means the other thing. Two reasons for it:
//!
//! **Layering.** Scripts have to run in the *simulation*, not in the editor.
//! §21's whole warning is about two front-ends that drift, and a script that
//! only ran when a GUI was open would be exactly that — a `.fusion` file whose
//! behaviour depended on which program opened it. So the interpreter has to
//! live at L2, next to the maths, and everything at L4 and below in this
//! workspace has no external dependencies at all. Linking CPython or
//! RustPython into the solver would abandon that for a feature nobody needs
//! the whole of.
//!
//! **Determinism.** §22 makes reproducibility a guarantee rather than an
//! aspiration. A language with no I/O, no allocation surprises, no iteration
//! over hash maps and a hard instruction budget is one whose output is a
//! function of the scene file. Arbitrary Python is not, and no amount of
//! sandboxing makes it so.
//!
//! ## The shape of a script
//!
//! A script is a body of statements that runs **once per solver step**, with
//! the body it is attached to bound to `self`:
//!
//! | name | meaning |
//! |---|---|
//! | `t`, `dt`, `step` | simulated time, the step's length, the step number |
//! | `self.position`, `self.rotation` | where it is, as vectors (rotation in radians) |
//! | `self.velocity`, `self.spin` | how it is moving |
//! | `self.force`, `self.torque` | accumulators, cleared before each step |
//! | `self.mass` | read-only |
//! | `world.gravity` | the scene's gravity |
//!
//! Writing `self.force` asks the solver for something; writing `self.position`
//! tells it. The first works on any dynamic body. The second is only honest on
//! a **kinematic** one — a body the solver does not integrate — because
//! setting the position of a body the solver is also carrying means two things
//! are deciding where it is, which is the drift §21 is about. The editor
//! changes a scripted body's type for you and says so.

mod builtin;
mod eval;
mod lex;
mod parse;

use std::collections::HashMap;
use std::fmt;

use fusion_math::Vec3;

pub use builtin::NAMES;
pub use eval::BUDGET;
pub use parse::MAX_NESTING;

/// Everything a script can hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Number(f32),
    Bool(bool),
    Vector(Vec3),
    /// Python's `None`.
    Nothing,
}

impl Value {
    /// The number in it, where there is one.
    pub fn as_number(self) -> Option<f32> {
        match self {
            Value::Number(n) => Some(n),
            Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// The vector in it, where there is one.
    pub fn as_vector(self) -> Option<Vec3> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }
}

/// What went wrong, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptError {
    /// One-based line in the script.
    pub line: u32,
    pub message: String,
}

impl ScriptError {
    pub fn new(line: u32, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ScriptError {}

/// What a script is allowed to see and touch.
///
/// The interpreter knows nothing about bodies. It asks for `self.position` by
/// name and is handed a [`Value`], which is what keeps this crate at L2 with
/// no dependency on `fusion-core` — and what lets the same language be pointed
/// at something else later without being rewritten.
pub trait Host {
    /// True where `name` is an object with fields, like `self`. Distinguishes
    /// `self.position` (a lookup) from `v.x` (a component of a value).
    fn is_object(&self, name: &str) -> bool;

    /// Read a field of an object.
    fn get(&self, object: &str, field: &str) -> Option<Value>;

    /// Write one. `Err` where the field is read-only or does not exist.
    fn set(&mut self, object: &str, field: &str, value: Value) -> Result<(), String>;

    /// Read a bare name the host supplies — `t`, `dt`, `step`.
    fn global(&self, name: &str) -> Option<Value>;
}

/// A compiled script.
///
/// Compiled once and run every step. Parsing a few lines is cheap, but doing
/// it sixty times a second per body would not be, and more to the point a
/// syntax error should be reported when the script is written rather than
/// discovered on every frame forever.
#[derive(Clone, Debug, Default)]
pub struct Program {
    body: Vec<parse::Line>,
    source: String,
}

impl Program {
    /// Compile source text.
    pub fn compile(source: &str) -> Result<Self, ScriptError> {
        let tokens = lex::scan(source)?;
        let body = parse::parse(&tokens)?;
        Ok(Self {
            body,
            source: source.to_string(),
        })
    }

    /// The text it was compiled from, so a scene can save what it loaded.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// True where there is nothing to run.
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Run one step against `host`, carrying `locals` between calls.
    pub fn run(
        &self,
        host: &mut dyn Host,
        locals: &mut HashMap<String, Value>,
    ) -> Result<(), ScriptError> {
        if self.body.is_empty() {
            return Ok(());
        }
        eval::Machine::new(host, locals).run(&self.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host with one object and a couple of globals, for testing the
    /// language rather than the engine.
    #[derive(Default)]
    struct Bench {
        fields: HashMap<String, Value>,
        time: f32,
    }

    impl Host for Bench {
        fn is_object(&self, name: &str) -> bool {
            name == "self"
        }
        fn get(&self, object: &str, field: &str) -> Option<Value> {
            (object == "self").then(|| self.fields.get(field).copied())?
        }
        fn set(&mut self, object: &str, field: &str, value: Value) -> Result<(), String> {
            if object != "self" {
                return Err(format!("no object called {object}"));
            }
            if field == "mass" {
                return Err("mass is read-only".into());
            }
            self.fields.insert(field.to_string(), value);
            Ok(())
        }
        fn global(&self, name: &str) -> Option<Value> {
            match name {
                "t" => Some(Value::Number(self.time)),
                "dt" => Some(Value::Number(1.0 / 60.0)),
                _ => None,
            }
        }
    }

    fn run(source: &str, host: &mut Bench) -> Result<HashMap<String, Value>, ScriptError> {
        let program = Program::compile(source)?;
        let mut locals = HashMap::new();
        program.run(host, &mut locals)?;
        Ok(locals)
    }

    /// The example from the module docs actually works.
    #[test]
    fn a_body_can_be_put_on_a_wave() {
        let mut host = Bench {
            fields: HashMap::from([("position".to_string(), Value::Vector(Vec3::ZERO))]),
            time: 0.25,
        };
        run("self.position.y = 2 + sin(t * 3) * 0.5", &mut host).expect("runs");
        let position = host.fields["position"].as_vector().expect("a vector");
        let expected = 2.0 + (0.25f32 * 3.0).sin() * 0.5;
        assert!((position.y - expected).abs() < 1e-6, "{position:?}");
        // And it touched nothing else.
        assert_eq!(position.x, 0.0);
        assert_eq!(position.z, 0.0);
    }

    #[test]
    fn variables_survive_between_steps() {
        let program =
            Program::compile("phase = phase + dt\nself.force.y = phase").expect("compiles");
        let mut host = Bench {
            fields: HashMap::from([("force".to_string(), Value::Vector(Vec3::ZERO))]),
            ..Default::default()
        };
        let mut locals = HashMap::from([("phase".to_string(), Value::Number(0.0))]);
        for _ in 0..3 {
            program.run(&mut host, &mut locals).expect("runs");
        }
        let phase = locals["phase"].as_number().expect("a number");
        assert!((phase - 3.0 / 60.0).abs() < 1e-6, "{phase}");
    }

    #[test]
    fn branches_choose() {
        let mut host = Bench {
            time: 5.0,
            ..Default::default()
        };
        run(
            "if t > 3:\n    x = 1\nelif t > 1:\n    x = 2\nelse:\n    x = 3\n",
            &mut host,
        )
        .map(|locals| assert_eq!(locals["x"], Value::Number(1.0)))
        .expect("runs");

        host.time = 2.0;
        let locals = run(
            "if t > 3:\n    x = 1\nelif t > 1:\n    x = 2\nelse:\n    x = 3\n",
            &mut host,
        )
        .expect("runs");
        assert_eq!(locals["x"], Value::Number(2.0));
    }

    /// The guard that keeps a bad script from taking the editor with it.
    #[test]
    fn a_loop_that_never_ends_is_stopped() {
        let mut host = Bench::default();
        let error = run("while True:\n    x = 1\n", &mut host).expect_err("should not finish");
        assert!(error.message.contains("never ends"), "{error}");
    }

    /// A budget spent across many steps must not accumulate: each step gets a
    /// fresh one, or a long-running script eventually fails for no reason.
    #[test]
    fn the_budget_resets_every_step() {
        let program = Program::compile("x = 0\nwhile x < 100:\n    x = x + 1\n").expect("compiles");
        let mut host = Bench::default();
        let mut locals = HashMap::new();
        for step in 0..50 {
            program
                .run(&mut host, &mut locals)
                .unwrap_or_else(|e| panic!("step {step}: {e}"));
        }
    }

    #[test]
    fn vectors_do_vector_things() {
        let mut host = Bench {
            fields: HashMap::from([(
                "velocity".to_string(),
                Value::Vector(Vec3::new(3.0, 4.0, 0.0)),
            )]),
            ..Default::default()
        };
        let locals = run(
            "speed = self.velocity.length()\nheading = self.velocity.normalized()\nn = dot(heading, heading)",
            &mut host,
        )
        .expect("runs");
        assert_eq!(locals["speed"], Value::Number(5.0));
        assert!((locals["n"].as_number().expect("number") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_read_only_field_says_so() {
        let mut host = Bench::default();
        let error = run("self.mass = 5", &mut host).expect_err("should refuse");
        assert!(error.message.contains("read-only"), "{error}");
    }

    #[test]
    fn an_unknown_name_names_itself() {
        let mut host = Bench::default();
        let error = run("x = wobble * 2", &mut host).expect_err("should refuse");
        assert!(error.message.contains("wobble"), "{error}");
        assert_eq!(error.line, 1);
    }

    /// The same script, the same inputs, the same numbers — the property §22
    /// makes a guarantee and everything else here exists to protect.
    #[test]
    fn the_same_script_gives_the_same_answer() {
        let source = "a = sin(t) * 3\nb = a * a + cos(t / 2)\nself.force.y = b\n";
        let program = Program::compile(source).expect("compiles");

        let sample = |seed: f32| {
            let mut host = Bench {
                fields: HashMap::from([("force".to_string(), Value::Vector(Vec3::ZERO))]),
                time: seed,
            };
            let mut locals = HashMap::new();
            program.run(&mut host, &mut locals).expect("runs");
            host.fields["force"].as_vector().expect("a vector").y
        };
        for step in 0..64 {
            let t = step as f32 * 0.01;
            assert_eq!(sample(t).to_bits(), sample(t).to_bits(), "at t = {t}");
        }
    }

    /// Nothing in the language reaches outside itself. If this list ever grows
    /// a way to open a file, the crate has stopped being what its docs say.
    #[test]
    fn there_is_no_way_out() {
        let mut host = Bench::default();
        for forbidden in [
            "import os",
            "open('x')",
            "__import__('os')",
            "exec('x')",
            "eval('x')",
            "print('x')",
        ] {
            assert!(
                run(forbidden, &mut host).is_err(),
                "{forbidden:?} was allowed"
            );
        }
    }

    /// Deep nesting is an error, not a crash.
    ///
    /// Recursive descent with no limit overflowed the stack at about ten
    /// thousand brackets, and a stack overflow aborts the process — so a
    /// corrupted `.fusion` file took the editor down on open. Each shape here
    /// recurses through a different path in the parser.
    #[test]
    fn nesting_too_deep_is_refused_rather_than_crashing() {
        let deep = 100_000;
        let cases = [
            format!("x = {}1{}", "(".repeat(deep), ")".repeat(deep)),
            format!("x = {}1", "-".repeat(deep)),
            format!("x = 2{}", " ** 2".repeat(deep)),
            format!("x = {}1", "not ".repeat(deep)),
            format!("x = {}1{}", "abs(".repeat(deep), ")".repeat(deep)),
        ];
        for source in &cases {
            let error = match Program::compile(source) {
                Ok(_) => panic!("{} compiled", &source[..12]),
                Err(error) => error,
            };
            assert!(
                error.message.contains("nests more than"),
                "{}: {}",
                &source[..12],
                error.message
            );
        }
        // And anything a person would write still compiles.
        let fine = 40;
        assert!(
            Program::compile(&format!("x = {}1{}", "(".repeat(fine), ")".repeat(fine))).is_ok()
        );
        assert!(Program::compile(&format!("x = {}1", "-".repeat(fine))).is_ok());
    }
}
