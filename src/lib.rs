//! A library for creating, using, and modifying artificial neural networks using the [Common
//! Genetic Encoding (CGE)][0]. See [`const_cge`][1] for a similar library geared towards embedded
//! environments and performance-critical use cases. For the creation of CGE-compatible neural
//! networks, see the [`eant2`][2] library.
//!
//! # Quick Start
//!
//! To load and use an existing neural network from a file:
//!
//! ```no_run
//! use cge::{Network, WithRecurrentState};
//!
//! // `extra` is any user-defined data stored alongside the network
//! let (mut network, metadata, extra) =
//!     Network::<f64>::load_file::<(), _>("network.cge", WithRecurrentState(true)).unwrap();
//!
//! println!("description: {:?}", metadata.description);
//! println!("num inputs, outputs: {}, {}", network.num_inputs(), network.num_outputs());
//! println!("{:?}", network.evaluate(&[1.0, 2.0]).unwrap());
//!
//! network.clear_state();
//!
//! println!("{:?}", network.evaluate(&[2.0, 0.0]).unwrap());
//! ```
//!
//! See [`Network`] for full documentation.
//!
//! [0]: https://dl.acm.org/doi/10.1145/1276958.1277162
//! [1]: https://github.com/wbrickner/const_cge
//! [2]: https://github.com/pengowen123/eant2

pub mod activation;
#[cfg(feature = "serde")]
pub mod encoding;
pub mod gene;
pub mod network;
mod stack;

pub use self::activation::Activation;
#[cfg(feature = "serde")]
pub use self::encoding::WithRecurrentState;
pub use self::network::Network;
