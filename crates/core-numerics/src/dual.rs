//! Forward-mode automatic differentiation: a number that carries its
//! gradient with respect to `N` inputs. For prototyping a new energy — write
//! it once on `Dual`, get its exact gradient — and for checking hand-derived
//! derivatives.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub value: f64,
    pub gradient: [f64; N],
}

impl<const N: usize> Dual<N> {
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.0; N],
        }
    }

    /// The `i`-th input, with value `value`.
    pub fn variable(value: f64, i: usize) -> Self {
        let mut gradient = [0.0; N];
        gradient[i] = 1.0;
        Self { value, gradient }
    }

    /// All `N` inputs at once.
    pub fn variables(values: [f64; N]) -> [Self; N] {
        std::array::from_fn(|i| Self::variable(values[i], i))
    }

    /// `f(self)` given `f(value)` and `f'(value)`.
    #[inline]
    fn chain(self, value: f64, derivative: f64) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|g| g * derivative),
        }
    }

    pub fn ln(self) -> Self {
        self.chain(self.value.ln(), 1.0 / self.value)
    }

    pub fn exp(self) -> Self {
        let e = self.value.exp();
        self.chain(e, e)
    }

    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        self.chain(s, 0.5 / s)
    }

    pub fn powi(self, n: i32) -> Self {
        self.chain(self.value.powi(n), n as f64 * self.value.powi(n - 1))
    }

    pub fn powf(self, p: f64) -> Self {
        self.chain(self.value.powf(p), p * self.value.powf(p - 1.0))
    }

    pub fn sin(self) -> Self {
        self.chain(self.value.sin(), self.value.cos())
    }

    pub fn cos(self) -> Self {
        self.chain(self.value.cos(), -self.value.sin())
    }

    pub fn abs(self) -> Self {
        self.chain(self.value.abs(), self.value.signum())
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            value: self.value + o.value,
            gradient: std::array::from_fn(|i| self.gradient[i] + o.gradient[i]),
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            value: self.value - o.value,
            gradient: std::array::from_fn(|i| self.gradient[i] - o.gradient[i]),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)] // the product rule
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            value: self.value * o.value,
            gradient: std::array::from_fn(|i| {
                self.gradient[i] * o.value + o.gradient[i] * self.value
            }),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)] // the quotient rule
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.value;
        Self {
            value: self.value * inv,
            gradient: std::array::from_fn(|i| {
                (self.gradient[i] - self.value * inv * o.gradient[i]) * inv
            }),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            gradient: self.gradient.map(|g| -g),
        }
    }
}

macro_rules! scalar_ops {
    ($($trait:ident $method:ident),*) => {$(
        impl<const N: usize> $trait<f64> for Dual<N> {
            type Output = Self;
            #[inline]
            fn $method(self, o: f64) -> Self {
                self.$method(Self::constant(o))
            }
        }
        impl<const N: usize> $trait<Dual<N>> for f64 {
            type Output = Dual<N>;
            #[inline]
            fn $method(self, o: Dual<N>) -> Dual<N> {
                Dual::constant(self).$method(o)
            }
        }
    )*};
}
scalar_ops!(Add add, Sub sub, Mul mul, Div div);
