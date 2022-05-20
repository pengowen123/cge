//! The error type for creation of networks.

use std::{error, fmt};

use crate::gene::NeuronId;

/// The reason why a genome is invalid.
#[derive(Clone, Debug)]
pub enum Error {
    /// The genome is empty.
    EmptyGenome,
    /// A neuron has an input count of zero. Contains the index and ID of the neuron gene.
    InvalidInputCount(usize, NeuronId),
    /// A neuron does not receive enough inputs. Contains the index and ID of the neuron gene.
    NotEnoughInputs(usize, NeuronId),
    /// Two or more neurons share the same ID. Contains the indices of the duplicates and their ID.
    DuplicateNeuronId(usize, usize, NeuronId),
    /// A non-neuron gene is an output of the network. Contains the index of the gene.
    NonNeuronOutput(usize),
    /// A forward jumper connection's parent neuron does not have a lesser depth than its source
    /// neuron. Contains the index of the forward jumper gene.
    InvalidForwardJumper(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::EmptyGenome => write!(f, "empty genome"),
            Self::InvalidInputCount(index, id) => write!(
                f,
                "invalid input count for neuron {} at index {}",
                id.as_usize(),
                index
            ),
            Self::NotEnoughInputs(index, id) => write!(
                f,
                "not enough inputs to neuron {} at index {}",
                id.as_usize(),
                index
            ),
            Self::DuplicateNeuronId(index_a, index_b, id) => write!(
                f,
                "duplicate neuron ID {} at indices {} and {}",
                id.as_usize(),
                index_a,
                index_b
            ),
            Self::NonNeuronOutput(index) => {
                write!(f, "non-neuron gene at index {} is a network output", index)
            }
            Self::InvalidForwardJumper(index) => {
                write!(f, "invalid forward jumper connection at index {}", index)
            }
        }
    }
}

impl error::Error for Error {}
