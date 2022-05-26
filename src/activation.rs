//! Handling of neuron activation functions.

use num_traits::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents which activation function to use when evaluating neurons.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Activation {
    /// Identity function. Outputs `x`.
    Linear,
    /// Heaviside or unit step function. Outputs `1` for `x > 0`, or `0` otherwise.
    UnitStep,
    /// Sign function. Outputs `1` for `x > 0`, `0` for `x = 0`, or `-1` otherwise.
    Sign,
    /// Logistic function. Outputs `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent function. Outputs `tanh(x)`.
    Tanh,
    /// Softsign function. Outputs `x / (1 + abs(x))`.
    SoftSign,
    /// Bent identity function. Outputs `(sqrt(x^2 + 1) - 1) / 2 + x`.
    BentIdentity,
    /// Rectified linear unit. Outputs `max(x, 0)`.
    Relu,
}

impl Activation {
    /// Applies the activation function to the input.
    pub fn apply<T: Float>(&self, x: T) -> T {
        match self {
            Activation::Linear => linear(x),
            Activation::UnitStep => unit_step(x),
            Activation::Sign => sign(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::Tanh => tanh(x),
            Activation::SoftSign => soft_sign(x),
            Activation::BentIdentity => bent_identity(x),
            Activation::Relu => relu(x),
        }
    }

    /// Returns the corresponding function to the `Activation`.
    pub fn get_function<T: Float>(&self) -> fn(T) -> T {
        match self {
            Activation::Linear => linear,
            Activation::UnitStep => unit_step,
            Activation::Sign => sign,
            Activation::Sigmoid => sigmoid,
            Activation::Tanh => tanh,
            Activation::SoftSign => soft_sign,
            Activation::BentIdentity => bent_identity,
            Activation::Relu => relu,
        }
    }
}

/// Outputs `x`.
pub fn linear<T>(x: T) -> T {
    x
}

/// Heaviside/unit step function. Outputs `1` for `x > 0`, or `0` otherwise.
pub fn unit_step<T: Float>(x: T) -> T {
    if x > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

/// Outputs `1` for `x > 0`, `0` for `x = 0`, or `-1` otherwise.
pub fn sign<T: Float>(x: T) -> T {
    if x > T::zero() {
        T::one()
    } else if x == T::zero() {
        T::zero()
    } else {
        -T::one()
    }
}

/// Logistic function. Outputs `1 / (1 + exp(-x))`.
pub fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// Outputs `tanh(x)`.
pub fn tanh<T: Float>(x: T) -> T {
    x.tanh()
}

/// Outputs `x / (1 + abs(x))`.
pub fn soft_sign<T: Float>(x: T) -> T {
    x / (T::one() + x.abs())
}

/// Outputs `(sqrt(x^2 + 1) - 1) / 2 + x`.
pub fn bent_identity<T: Float>(x: T) -> T {
    (((x.powi(2) + T::one()).sqrt() - T::one()) / (T::one() + T::one())) + x
}

/// Rectified linear unit. Outputs `max(0, x)`.
pub fn relu<T: Float>(x: T) -> T {
    x.max(T::zero())
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_activation() {
        assert_approx_eq!(5.0, Activation::Linear.apply(5.0));
        assert_approx_eq!(0.0, Activation::UnitStep.apply(-5.0));
        assert_approx_eq!(-1.0, Activation::Sign.apply(-5.0));
        assert_approx_eq!(0.8807970779778823, Activation::Sigmoid.apply(2.0));
        assert_approx_eq!(0.9640275800758169, Activation::Tanh.apply(2.0));
        assert_approx_eq!(0.8333333333333334, Activation::SoftSign.apply(5.0));
        assert_approx_eq!(7.049509756796392, Activation::BentIdentity.apply(5.0));
        assert_approx_eq!(5.0, Activation::Relu.apply(5.0));
        assert_approx_eq!(0.0, Activation::Relu.apply(-5.0));
    }
}
