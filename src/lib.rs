//! An implementation of the CGE neural network encoding. The Network struct has methods for
//! evaluating a neural network, resetting its state, and saving to and loading from files.
//!
//! # Examples
//!
<<<<<<< HEAD
//! ```no_run
=======
//! ```
>>>>>>> 9f469de66585ebab2be137a99e708cbbbeb27db3
//! use cge::Network;
//!
//! // Load a neural network from a file
//! let mut network = Network::load_from_file("neural_network.ann").unwrap();
//! 
//! // Get the output of the neural network with the specified inputs
<<<<<<< HEAD
//! let result = network.evaluate(&vec![1.0, 1.0]);
=======
//! let result = network.evaluate(vec![1.0, 1.0]);
>>>>>>> 9f469de66585ebab2be137a99e708cbbbeb27db3
//!
//! // Reset the state of the neural network
//! network.clear_state();
//! ```

// If values are too small/large, it shouldn't cause any problems. Operations on non-normal numbers
// result in non-normal numbers, never panicking.

// TODO: Add tests for network evaluation and file operations

mod utils;
mod file;
pub mod gene;
pub mod network;

pub use self::network::Network;
