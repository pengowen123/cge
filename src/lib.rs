//! An implementation of the CGE neural network encoding. The Network struct has methods for
//! evaluating a neural network, resetting its state, and saving to and loading from files and
//! strings.
//!
//! # Examples
//!
//! ```no_run
//! use cge::Network;
//!
//! // Load a neural network from a file
//! let mut network = Network::load_from_file("neural_network.ann").unwrap();
//! 
//! // Get the output of the neural network with the specified inputs
//! let result = network.evaluate(&vec![1.0, 1.0]);
//!
//! // Reset the state of the neural network
//! network.clear_state();
//! ```

// If values are too small/large, it shouldn't cause any problems. Operations on non-normal numbers
// result in non-normal numbers, never panicking.

// TODO: Implement a display for Network (make a pretty tree of lines)
// In the future, create a program for visualizing a neural network (generate an image or html)

mod utils;
mod file;
pub mod activation;
pub mod gene;
pub mod network;

pub use self::network::Network;
pub use self::activation::Activation;
