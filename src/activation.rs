//! Handling of neuron activation functions.

/// Represents which activation function to use when evaluating neurons.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
    pub fn apply(&self, x: f64) -> f64 {
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
    pub fn get_function(&self) -> fn(f64) -> f64 {
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
pub fn linear(x: f64) -> f64 {
    x
}

/// Heaviside/unit step function. Outputs `1` for `x > 0`, or `0` otherwise.
pub fn unit_step(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Outputs `1` for `x > 0`, `0` for `x = 0`, or `-1` otherwise.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x == 0.0 {
        0.0
    } else {
        -1.0
    }
}

/// Logistic function. Outputs `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Outputs `tanh(x)`.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Outputs `x / (1 + abs(x))`.
pub fn soft_sign(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// Outputs `(sqrt(x^2 + 1) - 1) / 2 + x`.
pub fn bent_identity(x: f64) -> f64 {
    (((x.powi(2) + 1.0).sqrt() - 1.0) / 2.0) + x
}

/// Rectified linear unit. Outputs `max(0, x)`.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}
