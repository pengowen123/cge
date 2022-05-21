//! The error type for creation of networks.

use std::num::TryFromIntError;
use std::{error, fmt};

use crate::gene::NeuronId;

/// The reason why a genome is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// The source neuron of a forward jumper or recurrent jumper gene does not exist. Contains the
    /// index of the jumper gene and the invalid neuron ID.
    InvalidJumperSource(usize, NeuronId),
    /// A forward jumper connection's parent neuron does not have a lesser depth than its source
    /// neuron. Contains the index of the forward jumper gene.
    InvalidForwardJumper(usize),
    /// An arithmetic operation or conversion overflowed or underflowed while building the network.
    Arithmetic,
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
            Self::InvalidJumperSource(index, source_id) => {
                write!(
                    f,
                    "jumper gene at index {} points to invalid neuron ID {}",
                    index,
                    source_id.as_usize()
                )
            }
            Self::InvalidForwardJumper(index) => {
                write!(f, "invalid forward jumper connection at index {}", index)
            }
            Self::Arithmetic => {
                write!(f, "integer overflow/underflow")
            }
        }
    }
}

impl error::Error for Error {}

impl From<TryFromIntError> for Error {
    fn from(_: TryFromIntError) -> Self {
        Self::Arithmetic
    }
}
