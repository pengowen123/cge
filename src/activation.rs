// For months the missing transfer function went unnoticed. I only realized this after failing to
// build a neural network to do XOR logic.

//! Option type for setting the transfer function.

/// Represents which transfer function to use for evaluating neural networks.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Activation {
    /// Maps input to output directly, as if there is no transfer function.
    Linear,
    /// Outputs 1 if input is greater than 0, 0 otherwise.
    Threshold,
    /// Outputs 1 if input is greater than 0, 0 if input is equal to 0, -1 otherwise. Useful
    /// for simple problems and boolean logic, as it only allows three possible output values.
    Sign,
    /// A non-linear function. This function is the most general, so it should be defaulted to.
    Sigmoid
}

pub fn threshold(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x == 0.0 {
        0.0
    } else {
        -1.0
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
