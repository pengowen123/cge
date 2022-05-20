//! An implementation of the CGE neural network encoding. The Network struct has methods for
//! evaluating a neural network, resetting its state, and saving to and loading from files and
//! strings.

// TODO: Implement a display for Network (make a pretty tree of lines)
// In the future, create a program for visualizing a neural network (generate an image or html)

pub mod activation;
pub mod encoding;
pub mod gene;
pub mod network;
mod stack;

pub use self::activation::Activation;
pub use self::network::Network;
