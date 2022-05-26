//! Error types related to networks.

use std::error;
use std::fmt::{self, Debug};
use std::num::TryFromIntError;

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

/// The reason why a mutation is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MutationError {
    /// A mutation adds no non-neuron genes.
    Empty,
    /// The parent neuron with the given ID does not exist.
    InvalidParent,
    /// The source neuron of a forward jumper or recurrent jumper gene does not exist.
    InvalidJumperSource,
    /// A forward jumper connection's parent neuron does not have a lesser depth than its source
    /// neuron.
    InvalidForwardJumper,
    /// The index of a gene removal is out of bounds.
    RemoveInvalidIndex,
    /// Attempted to remove a [`Neuron`][crate::gene::Neuron] gene.
    RemoveNeuron,
    /// Attempted to remove the only incoming connection of a [`Neuron`][crate::gene::Neuron] gene.
    RemoveOnlyInput,
    /// An arithmetic operation or conversion overflowed or underflowed while performing the
    /// mutation.
    Arithmetic,
}

impl fmt::Display for MutationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "mutation contains no non-neuron genes"),
            Self::InvalidParent => write!(f, "invalid parent neuron ID"),
            Self::InvalidJumperSource => write!(f, "jumper gene points to invalid neuron ID"),
            Self::InvalidForwardJumper => write!(f, "invalid forward jumper connection"),
            Self::RemoveInvalidIndex => write!(f, "removal index out of bounds"),
            Self::RemoveNeuron => write!(f, "cannot remove a neuron gene"),
            Self::RemoveOnlyInput => {
                write!(f, "cannot remove only incoming connection gene of a neuron")
            }
            Self::Arithmetic => write!(f, "integer overflow/underflow"),
        }
    }
}

impl error::Error for MutationError {}

/// An array of an incorrect length was passed to a [`Network`][crate::Network] method.
#[derive(Clone, Debug)]
pub struct MismatchedLengthsError;

impl fmt::Display for MismatchedLengthsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "length of source array does not equal length of target array"
        )
    }
}

impl error::Error for MismatchedLengthsError {}

/// The index was out of bounds.
#[derive(Clone, Debug)]
pub struct IndexOutOfBoundsError;

impl fmt::Display for IndexOutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "index out of bounds")
    }
}

impl error::Error for IndexOutOfBoundsError {}

/// Too few inputs were passed to [`Network::evaluate`][crate::Network].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotEnoughInputsError {
    expected: usize,
    provided: usize,
}

impl NotEnoughInputsError {
    pub(crate) fn new(expected: usize, provided: usize) -> Self {
        Self { expected, provided }
    }
}

impl fmt::Display for NotEnoughInputsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "too few inputs to a network: expected {}, provided {}",
            self.expected, self.provided
        )
    }
}

impl error::Error for NotEnoughInputsError {}
