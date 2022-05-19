//! The neural network struct.

use std::collections::HashMap;
use std::io;
use std::ops::Range;

use crate::activation::*;
use crate::evaluate::{self, Inputs};
use crate::file;
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
    /// If too little inputs are given, the empty inputs will have a value of zero. If too many
    /// inputs are given, the extras are discarded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cge::Network;
    ///
    /// // Load a neural network
    /// let mut network = Network::load_from_file("neural_network.ann").unwrap();
    ///
    /// // Get the output of the neural network the the specified inputs
    /// let result = network.evaluate(&vec![1.0, 1.0]).unwrap();
    ///
    /// // Get the output of the neural network with no inputs
    /// let result = network.evaluate(&[]).unwrap();
    ///
    /// // Get the output of the neural network with too many inputs (extras aren't used)
    /// let result = network.evaluate(&[1.0, 1.0, 1.0]).unwrap();
    ///
    /// // Let's say adder.ann is a file with a neural network with recurrent connections, used for
    /// // adding numbers together.
    /// let mut adder = Network::load_from_file("adder.ann").unwrap();
    ///
    /// // result_one will be 1.0
    /// let result_one = adder.evaluate(&[1.0]).unwrap();
    ///
    /// // result_two will be 3.0
    /// let result_two = adder.evaluate(&[2.0]).unwrap();
    ///
    /// // result_three will be 5.0
    /// let result_three = adder.evaluate(&[2.0]).unwrap();
    ///
    /// // If this behavior is not desired, call the clear_state method between evaluations:
    /// let result_one = adder.evaluate(&[1.0]).unwrap();
    ///
    /// adder.clear_state();
    ///
    /// // The 1.0 from the previous call is gone, so result_two will be 2.0
    /// let result_two = adder.evaluate(&[2.0]).unwrap();
    /// ```
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
    /// This state is only used by [`RecurrentJumper`][gene::RecurrentJumper] connections, so
    /// calling this method is unnecessary if the network does not contain them.
    pub fn clear_state(&mut self) {
        for gene in &mut self.genome {
            if let Gene::Neuron(neuron) = gene {
                neuron.set_previous_value(0.0);
            }
        }
    }

    /// Loads a neural network from a string. Returns `None` if the format is incorrect.
    ///
    /// # Examples
    ///
    /// ```
    /// use cge::Network;
    ///
    /// // Store the neural network in a string
    /// let string = "0: n 1 0 2,n 1 1 2,n 1 3 2,
    ///               i 1 0,i 1 1,i 1 1,n 1 2 4,
    ///               f 1 3,i 1 0,i 1 1,r 1 0";
    ///
    /// // Load a neural network from the string
    /// let network = Network::from_str(string).unwrap();
    /// ```
    ///
    /// # Format
    ///
    /// The format for the string is simple enough to build by hand:
    ///
    /// First, the number 0, 1, 2 or 3 is entered to represent the linear, threshold, sign, or
    /// sigmoid function, followed by a colon. The rest is the genome, encoded with comma
    /// separated genes:
    ///
    /// Neuron:     n [weight] [id] [input count]
    /// Input:      i [weight] [id]
    /// Connection: f [weight] [id]
    /// Recurrent:  r [weight] [id]
    /// Bias:       b [weight]
    ///
    /// For more information about what this means, see [here][1].
    ///
    /// [1]: http://www.academia.edu/6923193/A_common_genetic_encoding_for_both_direct_and_indirect_encodings_of_networks
    pub fn from_str(string: &str) -> Option<Network> {
        file::from_str(string)
    }

    /// Saves the neural network to a string. Allows embedding a neural network in source code.
    pub fn to_str(&self) -> String {
        file::to_str(self)
    }

    /// Saves the neural network to a file. Returns an empty tuple on success, or an io error.
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        file::write_network(self, path)
    }

    /// Loads a neural network from a file. No guarantees are made about the validity of the
    /// genome. Returns the network, or an io error. If the file is in a bad format,
    /// `std::io::ErrorKind::InvalidData` is returned.
    pub fn load_from_file(path: &str) -> io::Result<Network> {
        file::read_network(path)
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

    // linear genome from fig. 5.3 in paper:
    // https://www.researchgate.net/profile/Yohannes_Kassahun/publication/266864021_Towards_a_Unified_Approach_to_Learning_and_Adaptation/links/54ba91790cf253b50e2d037d.pdf?origin=publication_detail
    const TEST_GENOME: &'static str = "0: n 0.6 0 2,n 0.8 1 2,n 0.9 3 2,i 0.1 0,i 0.4 1,i 0.5 1,n 0.2 2 4,f 0.3 3,i 0.7 0,i 0.8 1,r 0.2 0";

    // This genome has one more neuron than TEST_GENOME
    // It is placed between neuron 3 and input id 1 by splitting one connection into two
    // Therefore the Genome has one more gene
    // The link weight connecting neuron 3 and 4 is 0.2,
    // The link weight connecting neuron 4 and input 1 is 0.3
    // removed original connection gene from neuron 3 to input 1
    const TEST_GENOME_2: &'static str = "0: n 0.6 0 2,n 0.8 1 2,n 0.9 3 2,i 0.1 0,n 0.2 4 1,i 0.3 1,i 0.5 1,n 0.2 2 4,f 0.3 3,i 0.7 0,i 0.8 1,r 0.2 0";

    #[test]
    fn test_genome_is_correct() {
        let mut net = Network::from_str(TEST_GENOME).unwrap();
        let output = net.evaluate(&vec![1.0, 1.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.654);
    }

    #[test]
    fn jump_recurrent_is_correct() {
        let mut net = Network::from_str(TEST_GENOME_2).unwrap();
        let output = net.evaluate(&vec![1.0, 1.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.49488);
    }

    #[test]
    fn test_recurrent_previous_value() {
        let genome = "0: n 1.0 0 2,r 3.0 1,n 1.0 1 1,b 1.0";

        let mut net = Network::from_str(genome).unwrap();
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
