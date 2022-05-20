//! The neural network struct.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::{error, fmt};

use crate::activation::*;
use crate::encoding::{self, CommonMetadata, EncodingVersion, MetadataVersion, PortableCGE};
use crate::evaluate::{self, Inputs};
use crate::gene::*;
use crate::stack::Stack;

/// Info about a neuron in a genome.
#[derive(Clone, Debug)]
pub struct NeuronInfo {
    subgenome_range: Range<usize>,
    depth: usize,
}

impl NeuronInfo {
    fn new(subgenome_range: Range<usize>, depth: usize) -> Self {
        Self {
            subgenome_range,
            depth,
        }
    }

    /// Returns the index range of the subgenome of this neuron.
    pub fn subgenome_range(&self) -> Range<usize> {
        self.subgenome_range.clone()
    }

    /// Returns the depth of this neuron.
    ///
    /// This is the number of implicit (non-jumper) connections between this neuron and the
    /// corresponding output neuron.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// The reason why a genome is invalid.
#[derive(Clone, Debug)]
pub enum InvalidNetworkError {
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

impl fmt::Display for InvalidNetworkError {
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

impl error::Error for InvalidNetworkError {}

/// Too few inputs were passed to [`Network::evaluate`].
#[derive(Clone, Debug)]
pub struct NotEnoughInputsError;

#[derive(Clone, Debug)]
pub struct Network {
    // The genes of the network
    genome: Vec<Gene>,
    // The activation function to use for neuron outputs
    activation: Activation,
    // The ID to use for the next neuron added to the network
    next_neuron_id: usize,
    // Info about each neuron, updated when the genome is changed
    neuron_info: HashMap<NeuronId, NeuronInfo>,
    // The number of inputs required by the network
    num_inputs: usize,
    // The number of network outputs
    num_outputs: usize,
    // The stack used when evaluating the `Network`
    stack: Stack,
}

impl Network {
    pub fn new(genome: Vec<Gene>, activation: Activation) -> Result<Self, InvalidNetworkError> {
        let next_neuron_id = genome
            .iter()
            .filter_map(|g| {
                if let Gene::Neuron(neuron) = g {
                    Some(neuron.id().as_usize())
                } else {
                    None
                }
            })
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);

        let mut network = Self {
            genome,
            activation,
            next_neuron_id,
            neuron_info: HashMap::new(),
            num_inputs: 0,
            num_outputs: 0,
            stack: Stack::new(),
        };

        network.rebuild_network_metadata()?;

        Ok(network)
    }

    /// Loads a previously-saved network, its metadata, and the user-defined extra data from a
    /// string. If no extra data is present, `E` can be set to `()`.
    pub fn load_str<'a, E>(s: &'a str) -> Result<(Network, CommonMetadata, E), encoding::Error>
    where
        E: Deserialize<'a>,
    {
        encoding::load_str(s)
    }

    /// Loads a previously-saved network, its metadata, and the user-defined extra data from a file.
    /// If no extra data is present, `E` can be set to `()`.
    pub fn load_file<E, P>(path: P) -> Result<(Network, CommonMetadata, E), encoding::Error>
    where
        E: DeserializeOwned,
        P: AsRef<Path>,
    {
        encoding::load_file(path)
    }

    /// Saves this network, its metadata, and an arbitrary extra data type to a string. `()` can be
    /// used if storing extra data is not needed.
    ///
    /// Using [`Metadata`][encoding::Metadata] will automatically use the latest encoding version,
    /// but a specific `Metadata` type can be used to select a specific version instead.
    pub fn to_string<E, M>(&self, metadata: M, extra: E) -> Result<String, encoding::Error>
    where
        E: Serialize,
        M: MetadataVersion<E>,
    {
        encoding::to_string(self.to_serializable(metadata, extra))
    }


    /// Saves this network, its metadata, and an arbitrary extra data type to a file. `()` can be
    /// used if storing extra data is not needed.
    ///
    /// Using [`Metadata`][encoding::Metadata] will automatically use the latest encoding version,
    /// but a specific `Metadata` type can be used to select a specific version instead.
    ///
    /// Recursively creates missing directories if `create_dirs` is `true`.
    pub fn to_file<E, M, P>(
        &self,
        metadata: M,
        extra: E,
        path: P,
        create_dirs: bool,
    ) -> Result<(), encoding::Error>
    where
        E: Serialize,
        M: MetadataVersion<E>,
        P: AsRef<Path>,
    {
        encoding::to_file(self.to_serializable(metadata, extra), path, create_dirs)
    }

    /// Converts the network to a serializable format. This can be used to save it in a format other
    /// than JSON. See [`PortableCGE`] for deserialization from different formats.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cge::Network;
    /// # let (network, _, ()) =
    /// #     Network::load_file(format!(
    /// #         "{}/test_data/test_network_v1.cge",
    /// #         env!("CARGO_MANIFEST_DIR")
    /// #     )).unwrap();
    /// use cge::encoding::{Metadata, PortableCGE};
    ///
    /// let metadata = Metadata::new("a description".to_string());
    /// let extra = ();
    /// let serializable = network.to_serializable(metadata, extra);
    ///
    /// // Any format supported by `serde`` can be used here
    /// let string = serde_json::to_string(&serializable).unwrap();
    ///
    /// // Other formats can be used when deserializing as well
    /// let deserialized: PortableCGE<()> = serde_json::from_str(&string).unwrap();
    /// let (network, metadata, extra) = deserialized.build().unwrap();
    /// ```
    pub fn to_serializable<E, M>(&self, metadata: M, extra: E) -> PortableCGE<E>
    where
        M: MetadataVersion<E>,
    {
        M::Data::new(self, metadata, extra)
    }

    /// Rebuilds the internal [`NeuronInfo`] map and other network metadata and checks the validity
    /// of the genome.
    fn rebuild_network_metadata(&mut self) -> Result<(), InvalidNetworkError> {
        // O(n)
        if self.genome.is_empty() {
            return Err(InvalidNetworkError::EmptyGenome);
        }

        let mut counter = 0isize;
        let mut neuron_info: HashMap<NeuronId, NeuronInfo> = HashMap::new();
        // Represents a stack of the current subgenomes being traversed
        // The value at the top of the stack when encountering a gene is that gene's parent
        let mut stopping_points = Vec::new();
        // A list of (jumper index, parent depth, source id) to check the validity of all forward
        // jumpers after `neuron_info` is completed
        let mut forward_jumper_checks = Vec::new();
        let mut max_input_id = None;

        for (i, gene) in self.genome.iter().enumerate() {
            let depth = stopping_points.len();
            // Each gene produces one output
            counter += 1;

            if let Gene::Neuron(neuron) = gene {
                // Track the value of `counter` when encountering a new subgenome (neuron) so that
                // the end of the subgenome can be detected and handled
                // The subgenome's starting index and depth are also added
                stopping_points.push((counter, neuron.id(), i, depth));

                // All neurons must have at least one input
                if neuron.num_inputs() == 0 {
                    return Err(InvalidNetworkError::InvalidInputCount(i, neuron.id()));
                }

                // Neuron genes consume a number of the following outputs equal to their required
                // number of inputs
                counter -= neuron.num_inputs() as isize;
            } else {
                // Subgenomes can only end on non-neuron genes

                // Non-neuron genes must have a parent because they cannot be network outputs
                if stopping_points.is_empty() {
                    return Err(InvalidNetworkError::NonNeuronOutput(i));
                }

                // Add forward jumper info to be checked later
                if let Gene::ForwardJumper(forward) = gene {
                    let parent_depth = depth - 1;
                    forward_jumper_checks.push((i, parent_depth, forward.source_id()));
                }

                // Check if `counter` has returned to its value from when any subgenomes started
                while !stopping_points.is_empty() && stopping_points.last().unwrap().0 == counter {
                    let (_, id, start_index, depth) = stopping_points.pop().unwrap();

                    if let Some(existing) = neuron_info.get(&id) {
                        let existing_index = existing.subgenome_range().start;
                        return Err(InvalidNetworkError::DuplicateNeuronId(
                            existing_index,
                            start_index,
                            id,
                        ));
                    }

                    let subgenome_range = start_index..i + 1;
                    neuron_info.insert(id, NeuronInfo::new(subgenome_range, depth));
                }

                if let Gene::Input(input) = gene {
                    max_input_id = max_input_id
                        .or(Some(0))
                        .map(|max_id| max_id.max(input.id().as_usize()));
                }
            }
        }

        // If any subgenomes were not fully traversed, a neuron did not receive enough inputs
        if let Some(&(_, id, index, _)) = stopping_points.last() {
            return Err(InvalidNetworkError::NotEnoughInputs(index, id));
        }

        // Check that forward jumpers always connect parent neurons to source neurons of higher
        // depth
        for (jumper_index, parent_depth, source_id) in forward_jumper_checks {
            if parent_depth >= neuron_info[&source_id].depth() {
                return Err(InvalidNetworkError::InvalidForwardJumper(jumper_index));
            }
        }

        self.neuron_info = neuron_info;
        // + 1 because input IDs start at zero, 0 if no IDs were found
        self.num_inputs = max_input_id.map(|id| id + 1).unwrap_or(0);
        self.num_outputs = counter as usize;

        Ok(())
    }

    /// Evaluates the neural network with the given inputs, returning a vector of outputs. The
    /// encoding can encode recurrent connections and bias inputs, so an internal state is used. It
    /// is important to run the clear_state method before calling evaluate again, unless it is
    /// desired to allow data carry over from the previous evaluation, for example if the network
    /// is being used as a real time controller.
    ///
    /// If too many inputs are given, the extras are discarded.
    pub fn evaluate(&mut self, inputs: &[f64]) -> Result<&[f64], NotEnoughInputsError> {
        if inputs.len() < self.num_inputs {
            return Err(NotEnoughInputsError);
        }

        // Clear any previous network outputs
        self.stack.clear();

        let inputs = Inputs(inputs);
        let length = self.genome.len();
        evaluate::evaluate_slice(
            &mut self.genome,
            0..length,
            inputs,
            &mut self.stack,
            false,
            &self.neuron_info,
            self.activation,
        );

        // Perform post-evaluation updates/cleanup
        update_stored_values(&mut self.genome);

        Ok(self.stack.as_slice())
    }

    /// Clears the persistent state of the neural network.
    ///
    /// This state is only used by [`RecurrentJumper`] connections, so calling this method is
    /// unnecessary if the network does not contain them.
    pub fn clear_state(&mut self) {
        for gene in &mut self.genome {
            if let Gene::Neuron(neuron) = gene {
                neuron.set_previous_value(0.0);
            }
        }
    }

    /// Returns the genome of this `Network`.
    pub fn genome(&self) -> &[Gene] {
        &self.genome
    }

    /// Returns the activation function of this `Network`.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Returns the number of inputs required by this `Network`.
    ///
    /// This is equal to one plus the highest input ID among [`Input`] genes in the network, which
    /// means that any unused IDs in the range `0..num_inputs` will correspond to unused values in
    /// the input array to [`evaluate`][Self::evaluate].
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns the number of outputs produced by this `Network` when evaluated.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

/// Moves the current value stored in each neuron into its previous value.
fn update_stored_values(genome: &mut [Gene]) {
    for gene in genome {
        if let Gene::Neuron(neuron) = gene {
            neuron.set_previous_value(
                neuron
                    .current_value()
                    .expect("neuron's current value is not set"),
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn get_file_path(file_name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), file_name)
    }

    #[test]
    fn test_evaluate() {
        // Example network from the CGE paper
        let (mut net, _, ()) = Network::load_file(get_file_path("test_network_v1.cge")).unwrap();
        let output = net.evaluate(&vec![1.0, 1.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.654);
    }

    #[test]
    fn test_recurrent_previous_value() {
        let (mut net, _, ()) = Network::load_file(get_file_path("test_network_recurrent.cge")).unwrap();

        // The recurrent jumper reads a previous value of zero despite the neuron already being
        // evaluated by the time the jumper is reached
        let output = net.evaluate(&[]).unwrap().to_vec();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1.0);

        // The recurrent jumper now reads a non-zero previous value from the first evaluation
        let output2 = net.evaluate(&[]).unwrap();
        assert_eq!(output2.len(), 1);
        assert_eq!(output2[0], 4.0);
    }
}
