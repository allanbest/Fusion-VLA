//! Walking the tree.
//!
//! A tree-walking interpreter, not a bytecode VM. The scripts this runs are a
//! handful of lines each and the cost that matters is the physics around them;
//! a compiler would be more code to be wrong in, for a saving nobody would be
//! able to measure.
//!
//! ## The budget
//!
//! Every node evaluated spends one unit of a per-step budget, and running out
//! is an error rather than a hang. This is the only thing standing between a
//! `while True:` in somebody's script and an editor that has to be killed from
//! a terminal — and a physics editor is exactly the place where somebody will
//! write one, because a script that runs every step *looks* like a loop
//! already.

use std::collections::HashMap;

use fusion_math::Vec3;

use crate::parse::{Expr, Line, Stmt, Target};
use crate::{Host, ScriptError, Value};

/// How many nodes one step of one script may evaluate.
///
/// Generous for anything sane — the wave-paddle script in the docs spends
/// about forty — and small enough that a runaway loop is caught inside a
/// frame rather than after one.
pub const BUDGET: u32 = 20_000;

pub struct Machine<'a> {
    host: &'a mut dyn Host,
    /// Named values that survive between steps, so a script can carry state
    /// without inventing somewhere to put it.
    pub locals: &'a mut HashMap<String, Value>,
    budget: u32,
    line: u32,
}

impl<'a> Machine<'a> {
    pub fn new(host: &'a mut dyn Host, locals: &'a mut HashMap<String, Value>) -> Self {
        Self {
            host,
            locals,
            budget: BUDGET,
            line: 0,
        }
    }

    fn spend(&mut self) -> Result<(), ScriptError> {
        match self.budget.checked_sub(1) {
            Some(left) => {
                self.budget = left;
                Ok(())
            }
            None => Err(ScriptError::new(
                self.line,
                format!(
                    "this script did more than {BUDGET} things in one step — \
                     check for a loop that never ends"
                ),
            )),
        }
    }

    pub fn run(&mut self, program: &[Line]) -> Result<(), ScriptError> {
        for line in program {
            self.line = line.line;
            self.statement(&line.stmt)?;
        }
        Ok(())
    }

    fn block(&mut self, body: &[Stmt]) -> Result<(), ScriptError> {
        for stmt in body {
            self.statement(stmt)?;
        }
        Ok(())
    }

    fn statement(&mut self, stmt: &Stmt) -> Result<(), ScriptError> {
        self.spend()?;
        match stmt {
            Stmt::Pass => Ok(()),
            Stmt::Eval(expr) => self.expression(expr).map(|_| ()),
            Stmt::Assign(target, expr) => {
                let value = self.expression(expr)?;
                self.store(target, value)
            }
            Stmt::Update(target, operator, expr) => {
                let right = self.expression(expr)?;
                let left = self.load(target)?;
                let value = self.binary(operator, left, right)?;
                self.store(target, value)
            }
            Stmt::If(arms, otherwise) => {
                for (condition, body) in arms {
                    if self.expression(condition)?.truthy() {
                        return self.block(body);
                    }
                }
                self.block(otherwise)
            }
            Stmt::While(condition, body) => {
                while self.expression(condition)?.truthy() {
                    self.spend()?;
                    self.block(body)?;
                }
                Ok(())
            }
        }
    }

    // --- places ---------------------------------------------------------

    /// Read what a target currently holds.
    fn load(&mut self, target: &Target) -> Result<Value, ScriptError> {
        let mut value = self.name(&target.root)?;
        for field in &target.path {
            value = self.field(value, field)?;
        }
        Ok(value)
    }

    /// Write to a target.
    ///
    /// A path of more than one step is a read-modify-write: `self.position.y =
    /// 3` fetches the whole vector, changes one component, and puts it back.
    /// That is the only thing it can be, because a `Vec3` is a value and there
    /// is nothing to hand out a reference into.
    fn store(&mut self, target: &Target, value: Value) -> Result<(), ScriptError> {
        match target.path.len() {
            0 => {
                // A bare name is a local. Assigning to a host global would be
                // writing to `t`, which is not a thing that can be true.
                if self.host.is_object(&target.root) {
                    return Err(ScriptError::new(
                        self.line,
                        format!("{} is an object, not a value to assign to", target.root),
                    ));
                }
                self.locals.insert(target.root.clone(), value);
                Ok(())
            }
            1 => self
                .host
                .set(&target.root, &target.path[0], value)
                .map_err(|e| ScriptError::new(self.line, e)),
            2 => {
                let outer = self
                    .host
                    .get(&target.root, &target.path[0])
                    .ok_or_else(|| {
                        ScriptError::new(
                            self.line,
                            format!("{}.{} does not exist", target.root, target.path[0]),
                        )
                    })?;
                let Value::Vector(mut vector) = outer else {
                    return Err(ScriptError::new(
                        self.line,
                        format!(
                            "{}.{} is not a vector, so it has no components to set",
                            target.root, target.path[0]
                        ),
                    ));
                };
                let component = self.number(value)?;
                match target.path[1].as_str() {
                    "x" => vector.x = component,
                    "y" => vector.y = component,
                    "z" => vector.z = component,
                    other => {
                        return Err(ScriptError::new(
                            self.line,
                            format!("a vector has x, y and z, not {other:?}"),
                        ))
                    }
                }
                self.host
                    .set(&target.root, &target.path[0], Value::Vector(vector))
                    .map_err(|e| ScriptError::new(self.line, e))
            }
            _ => Err(ScriptError::new(
                self.line,
                "that is deeper than anything in this language goes",
            )),
        }
    }

    // --- expressions ----------------------------------------------------

    fn expression(&mut self, expr: &Expr) -> Result<Value, ScriptError> {
        self.spend()?;
        match expr {
            Expr::Number(value) => Ok(Value::Number(*value)),
            Expr::Bool(value) => Ok(Value::Bool(*value)),
            Expr::Nothing => Ok(Value::Nothing),
            Expr::Name(name) => self.name(name),
            Expr::Field(inner, field) => {
                // `self.position` is a host lookup; anything else is a field
                // on a value that has already been produced.
                if let Expr::Name(root) = inner.as_ref() {
                    if self.host.is_object(root) {
                        return self.host.get(root, field).ok_or_else(|| {
                            ScriptError::new(self.line, format!("{root} has no {field:?}"))
                        });
                    }
                }
                let value = self.expression(inner)?;
                self.field(value, field)
            }
            Expr::Call(name, arguments) => {
                let mut values = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    values.push(self.expression(argument)?);
                }
                crate::builtin::call(name, &values).map_err(|e| ScriptError::new(self.line, e))
            }
            Expr::Method(inner, name, arguments) => {
                let receiver = self.expression(inner)?;
                let mut values = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    values.push(self.expression(argument)?);
                }
                crate::builtin::method(receiver, name, &values)
                    .map_err(|e| ScriptError::new(self.line, e))
            }
            Expr::Unary(operator, inner) => {
                let value = self.expression(inner)?;
                match *operator {
                    "-" => match value {
                        Value::Number(n) => Ok(Value::Number(-n)),
                        Value::Vector(v) => Ok(Value::Vector(-v)),
                        other => Err(ScriptError::new(
                            self.line,
                            format!("{} cannot be negated", other.kind()),
                        )),
                    },
                    _ => Ok(Value::Bool(!value.truthy())),
                }
            }
            Expr::Binary(operator, left, right) => {
                // `and` and `or` short-circuit, which is why they cannot be
                // folded into the arithmetic below: the right side of an `and`
                // whose left is false must not be evaluated at all.
                match *operator {
                    "and" => {
                        let left = self.expression(left)?;
                        if !left.truthy() {
                            return Ok(left);
                        }
                        return self.expression(right);
                    }
                    "or" => {
                        let left = self.expression(left)?;
                        if left.truthy() {
                            return Ok(left);
                        }
                        return self.expression(right);
                    }
                    _ => {}
                }
                let a = self.expression(left)?;
                let b = self.expression(right)?;
                self.binary(operator, a, b)
            }
        }
    }

    fn name(&mut self, name: &str) -> Result<Value, ScriptError> {
        if let Some(value) = self.locals.get(name) {
            return Ok(*value);
        }
        if let Some(value) = self.host.global(name) {
            return Ok(value);
        }
        if let Some(value) = crate::builtin::constant(name) {
            return Ok(value);
        }
        Err(ScriptError::new(
            self.line,
            format!("nothing here is called {name:?}"),
        ))
    }

    fn field(&mut self, value: Value, field: &str) -> Result<Value, ScriptError> {
        match (&value, field) {
            (Value::Vector(v), "x") => Ok(Value::Number(v.x)),
            (Value::Vector(v), "y") => Ok(Value::Number(v.y)),
            (Value::Vector(v), "z") => Ok(Value::Number(v.z)),
            _ => Err(ScriptError::new(
                self.line,
                format!("{} has no {field:?}", value.kind()),
            )),
        }
    }

    fn number(&self, value: Value) -> Result<f32, ScriptError> {
        match value {
            Value::Number(n) => Ok(n),
            Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
            other => Err(ScriptError::new(
                self.line,
                format!("expected a number, found {}", other.kind()),
            )),
        }
    }

    fn binary(&mut self, operator: &str, a: Value, b: Value) -> Result<Value, ScriptError> {
        use Value::{Bool, Number, Vector};

        // Comparisons first: they answer for every pair of types, and folding
        // them in with arithmetic would mean `==` needing a case per
        // combination rather than one.
        match operator {
            "==" => return Ok(Bool(a.same_as(&b))),
            "!=" => return Ok(Bool(!a.same_as(&b))),
            "<" | ">" | "<=" | ">=" => {
                let (x, y) = (self.number(a)?, self.number(b)?);
                return Ok(Bool(match operator {
                    "<" => x < y,
                    ">" => x > y,
                    "<=" => x <= y,
                    _ => x >= y,
                }));
            }
            _ => {}
        }

        let value = match (a, b, operator) {
            (Number(x), Number(y), "+") => Number(x + y),
            (Number(x), Number(y), "-") => Number(x - y),
            (Number(x), Number(y), "*") => Number(x * y),
            (Number(x), Number(y), "/") => Number(x / y),
            (Number(x), Number(y), "%") => Number(x.rem_euclid(y)),
            (Number(x), Number(y), "//") => Number((x / y).floor()),
            (Number(x), Number(y), "**") => Number(x.powf(y)),

            (Vector(x), Vector(y), "+") => Vector(x + y),
            (Vector(x), Vector(y), "-") => Vector(x - y),
            // Component-wise, not a dot product. `dot()` is a function, and a
            // `*` between two vectors that quietly meant one of the two would
            // be a coin flip every time it was read.
            (Vector(x), Vector(y), "*") => Vector(x.mul_element_wise(y)),
            (Vector(v), Number(s), "*") | (Number(s), Vector(v), "*") => Vector(v * s),
            (Vector(v), Number(s), "/") => Vector(v * (1.0 / s)),

            (x, y, operator) => {
                return Err(ScriptError::new(
                    self.line,
                    format!(
                        "{} {operator} {} is not something this language can do",
                        x.kind(),
                        y.kind()
                    ),
                ))
            }
        };
        Ok(value)
    }
}

impl Value {
    /// What a condition makes of it.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0,
            Value::Vector(v) => *v != Vec3::ZERO,
            Value::Nothing => false,
        }
    }

    /// Equality across types: a number is never a vector.
    fn same_as(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Vector(a), Value::Vector(b)) => a == b,
            (Value::Nothing, Value::Nothing) => true,
            _ => false,
        }
    }

    /// The name of its type, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Number(_) => "a number",
            Value::Bool(_) => "a true/false",
            Value::Vector(_) => "a vector",
            Value::Nothing => "None",
        }
    }
}
