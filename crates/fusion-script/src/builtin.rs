//! The standard library, such as it is.
//!
//! Everything a script can call, and nothing else. There is no `import`, so
//! this list is the whole surface — which is the point: a script that can only
//! do arithmetic on the body it is attached to cannot read a file, cannot open
//! a socket, and cannot make a run irreproducible. Determinism (§22) is a
//! property of the scene, and a script is part of the scene only if the same
//! script produces the same numbers on every machine that runs it.
//!
//! Everything here is `f32` and IEEE-deterministic: the transcendental
//! functions are Rust's, which are the platform's, and those are *not*
//! guaranteed identical across targets. Same build, same machine, bit for bit
//! — which is what §22 actually promises — holds. Across two different
//! operating systems, a script using `sin` may diverge in the last bit, and
//! that is stated rather than hidden.

use fusion_math::Vec3;

use crate::Value;

/// `pi`, `tau`, `e`, and the two vectors worth having by name.
pub fn constant(name: &str) -> Option<Value> {
    Some(match name {
        "pi" => Value::Number(std::f32::consts::PI),
        "tau" => Value::Number(std::f32::consts::TAU),
        "e" => Value::Number(std::f32::consts::E),
        "up" => Value::Vector(Vec3::Y),
        "zero" => Value::Vector(Vec3::ZERO),
        _ => return None,
    })
}

/// Call a free function.
pub fn call(name: &str, arguments: &[Value]) -> Result<Value, String> {
    let number = |index: usize| -> Result<f32, String> {
        match arguments.get(index) {
            Some(Value::Number(n)) => Ok(*n),
            Some(Value::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
            Some(other) => Err(format!(
                "{name} wants a number for argument {}, and got {}",
                index + 1,
                other.kind()
            )),
            None => Err(format!(
                "{name} wants {} arguments and got {}",
                index + 1,
                arguments.len()
            )),
        }
    };

    let one = |f: fn(f32) -> f32| -> Result<Value, String> { Ok(Value::Number(f(number(0)?))) };

    Ok(match name {
        "sin" => one(f32::sin)?,
        "cos" => one(f32::cos)?,
        "tan" => one(f32::tan)?,
        "asin" => one(f32::asin)?,
        "acos" => one(f32::acos)?,
        "atan" => one(f32::atan)?,
        "sqrt" => one(f32::sqrt)?,
        "exp" => one(f32::exp)?,
        "log" => one(f32::ln)?,
        "floor" => one(f32::floor)?,
        "ceil" => one(f32::ceil)?,
        "round" => one(f32::round)?,
        "sign" => one(f32::signum)?,
        "radians" => one(f32::to_radians)?,
        "degrees" => one(f32::to_degrees)?,
        "atan2" => Value::Number(number(0)?.atan2(number(1)?)),
        "pow" => Value::Number(number(0)?.powf(number(1)?)),
        "abs" => match arguments.first() {
            Some(Value::Vector(v)) => Value::Number(v.length()),
            _ => one(f32::abs)?,
        },
        "min" => Value::Number(number(0)?.min(number(1)?)),
        "max" => Value::Number(number(0)?.max(number(1)?)),
        // Clamped with `max` then `min`, so a range given backwards returns
        // the low end rather than NaN. Somebody will type `clamp(x, 1, 0)`.
        "clamp" => Value::Number(number(0)?.max(number(1)?).min(number(2)?)),
        "lerp" => {
            let (a, b, t) = (number(0)?, number(1)?, number(2)?);
            Value::Number(a + (b - a) * t)
        }
        "vec" => Value::Vector(Vec3::new(number(0)?, number(1)?, number(2)?)),
        "dot" => match (arguments.first(), arguments.get(1)) {
            (Some(Value::Vector(a)), Some(Value::Vector(b))) => Value::Number(a.dot(*b)),
            _ => return Err("dot wants two vectors".into()),
        },
        "cross" => match (arguments.first(), arguments.get(1)) {
            (Some(Value::Vector(a)), Some(Value::Vector(b))) => Value::Vector(a.cross(*b)),
            _ => return Err("cross wants two vectors".into()),
        },
        _ => return Err(format!("there is no function called {name:?}")),
    })
}

/// Call a method on a value.
pub fn method(receiver: Value, name: &str, arguments: &[Value]) -> Result<Value, String> {
    let Value::Vector(vector) = receiver else {
        return Err(format!("{} has no methods", receiver.kind()));
    };
    Ok(match name {
        "length" => Value::Number(vector.length()),
        "normalized" => Value::Vector(vector.normalize_or_zero()),
        "dot" => match arguments.first() {
            Some(Value::Vector(other)) => Value::Number(vector.dot(*other)),
            _ => return Err("dot wants a vector".into()),
        },
        "cross" => match arguments.first() {
            Some(Value::Vector(other)) => Value::Vector(vector.cross(*other)),
            _ => return Err("cross wants a vector".into()),
        },
        _ => return Err(format!("a vector has no method called {name:?}")),
    })
}

/// Every name a script can use without defining it. What the editor offers as
/// completions, and what the docs list.
pub const NAMES: &[&str] = &[
    "abs", "acos", "asin", "atan", "atan2", "ceil", "clamp", "cos", "cross", "degrees", "dot", "e",
    "exp", "floor", "lerp", "log", "max", "min", "pi", "pow", "radians", "round", "sign", "sin",
    "sqrt", "tan", "tau", "up", "vec", "zero",
];
