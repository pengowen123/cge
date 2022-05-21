//! The neural network struct.

mod error;
mod evaluate;

pub use error::Error;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use crate::activation::*;
use crate::encoding::{self, CommonMetadata, EncodingVersion, MetadataVersion, PortableCGE};
use crate::gene::*;
use crate::stack::Stack;
use evaluate::Inputs;

/// Info about a neuron in a genome.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn new(genome: Vec<Gene>, activation: Activation) -> Result<Self, Error> {
        let mut network = Self {
            genome,
            activation,
            next_neuron_id: 0,
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
    /// // Any format supported by `serde` can be used here
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
    #[deny(clippy::integer_arithmetic, clippy::as_conversions)]
    fn rebuild_network_metadata(&mut self) -> Result<(), Error> {
        // O(n)
        if self.genome.is_empty() {
            return Err(Error::EmptyGenome);
        }

        let mut counter = 0isize;
        let mut neuron_info: HashMap<NeuronId, NeuronInfo> = HashMap::new();
        // Represents a stack of the current subgenomes being traversed
        // The value at the top of the stack when encountering a gene is that gene's parent
        let mut stopping_points = Vec::new();
        // A list of (jumper index, parent depth, source id) to check the validity of all forward
        // jumpers after `neuron_info` is completed
        let mut forward_jumper_checks = Vec::new();
        // A list of (jumper index, source id) to check the validity of all recurrent jumpers after
        // `neuron_info` is completed
        let mut recurrent_jumper_checks = Vec::new();
        let mut max_input_id = None;
        let mut max_neuron_id = None;

        for (i, gene) in self.genome.iter().enumerate() {
            let depth = stopping_points.len();
            // Each gene produces one output
            counter = counter.checked_add(1).ok_or(Error::Arithmetic)?;

            if let Gene::Neuron(neuron) = gene {
                // Track the value of `counter` when encountering a new subgenome (neuron) so that
                // the end of the subgenome can be detected and handled
                // The subgenome's starting index and depth are also added
                stopping_points.push((counter, neuron.id(), i, depth));

                // All neurons must have at least one input
                if neuron.num_inputs() == 0 {
                    return Err(Error::InvalidInputCount(i, neuron.id()));
                }

                // Neuron genes consume a number of the following outputs equal to their required
                // number of inputs
                let num_inputs = isize::try_from(neuron.num_inputs())?;
                counter = counter.checked_sub(num_inputs).ok_or(Error::Arithmetic)?;

                max_neuron_id = max_neuron_id
                    .or(Some(0))
                    .map(|max_id| max_id.max(neuron.id().as_usize()));
            } else {
                // Subgenomes can only end on non-neuron genes

                // Non-neuron genes must have a parent because they cannot be network outputs
                if stopping_points.is_empty() {
                    return Err(Error::NonNeuronOutput(i));
                }

                // Add jumper info to be checked later
                match gene {
                    Gene::ForwardJumper(forward) => {
                        let parent_depth = depth.checked_sub(1).ok_or(Error::Arithmetic)?;
                        forward_jumper_checks.push((i, parent_depth, forward.source_id()));
                    }
                    Gene::RecurrentJumper(recurrent) => {
                        recurrent_jumper_checks.push((i, recurrent.source_id()))
                    }
                    _ => {}
                }

                // Check if `counter` has returned to its value from when any subgenomes started
                while !stopping_points.is_empty() && stopping_points.last().unwrap().0 == counter {
                    let (_, id, start_index, depth) = stopping_points.pop().unwrap();

                    if let Some(existing) = neuron_info.get(&id) {
                        let existing_index = existing.subgenome_range().start;
                        return Err(Error::DuplicateNeuronId(existing_index, start_index, id));
                    }

                    let end_index = i.checked_add(1).ok_or(Error::Arithmetic)?;
                    let subgenome_range = start_index..end_index;
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
            return Err(Error::NotEnoughInputs(index, id));
        }

        // Check that forward jumpers always connect parent neurons to source neurons of higher
        // depth
        for (jumper_index, parent_depth, source_id) in forward_jumper_checks {
            if let Some(source_info) = neuron_info.get(&source_id) {
                if parent_depth >= source_info.depth() {
                    return Err(Error::InvalidForwardJumper(jumper_index));
                }
            } else {
                // Return an error if the jumper's source does not exist
                return Err(Error::InvalidJumperSource(jumper_index, source_id));
            }
        }

        // Check that the source of every recurrent jumper exists
        for (jumper_index, source_id) in recurrent_jumper_checks {
            if !neuron_info.contains_key(&source_id) {
                return Err(Error::InvalidJumperSource(jumper_index, source_id));
            }
        }

        self.neuron_info = neuron_info;
        // This unwrap is safe because genomes must have at least one neuron
        self.next_neuron_id = max_neuron_id
            .unwrap()
            .checked_add(1)
            .ok_or(Error::Arithmetic)?;
        // + 1 because input IDs start at zero, 0 if no IDs were found
        self.num_inputs = match max_input_id {
            Some(id) => id.checked_add(1).ok_or(Error::Arithmetic)?,
            None => 0,
        };
        // The validity checks above should guarantee the safety of this unwrap
        self.num_outputs = usize::try_from(counter).unwrap();

        Ok(())
    }

    /// Evaluates the neural network with the given inputs, returning a vector of outputs. The
    /// encoding can encode recurrent connections and bias inputs, so an internal state is used. It
    /// is important to run the clear_state method before calling evaluate again, unless it is
    /// desired to allow data carry over from the previous evaluation, for example if the network
    /// is being used as a real time controller.
    ///
    /// If too many inputs are given, the extras are discarded.
    pub fn evaluate(&mut self, inputs: &[f64]) -> Option<&[f64]> {
        if inputs.len() < self.num_inputs {
            return None;
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

        Some(self.stack.as_slice())
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

    /// Sets the activation function of this `Network`.
    pub fn set_activation(&mut self, new: Activation) {
        self.activation = new;
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
            neuron.set_current_value(None);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn get_file_path(file_name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), file_name)
    }

    fn bias() -> Gene {
        Bias::new(1.0).into()
    }

    fn input(id: usize) -> Gene {
        Input::new(InputId::new(id), 1.0).into()
    }

    fn neuron(id: usize, num_inputs: usize) -> Gene {
        Neuron::new(NeuronId::new(id), num_inputs, 1.0).into()
    }

    fn forward(source_id: usize) -> Gene {
        ForwardJumper::new(NeuronId::new(source_id), 1.0).into()
    }

    fn recurrent(source_id: usize) -> Gene {
        RecurrentJumper::new(NeuronId::new(source_id), 1.0).into()
    }

    fn check_num_outputs(network: &Network) {
        assert_eq!(
            network.num_outputs(),
            network
                .neuron_info
                .iter()
                .filter(|(_, info)| info.depth == 0)
                .count()
        );
    }

    #[test]
    fn test_inputs_outputs() {
        let genome = vec![neuron(0, 2), input(0), bias()];
        let net = Network::new(genome, Activation::Linear).unwrap();
        assert_eq!(1, net.num_inputs());
        assert_eq!(1, net.num_outputs());
        check_num_outputs(&net);

        let genome2 = vec![neuron(0, 3), input(0), bias(), input(2)];
        let net2 = Network::new(genome2, Activation::Linear).unwrap();
        assert_eq!(3, net2.num_inputs());
        assert_eq!(1, net2.num_outputs());
        check_num_outputs(&net2);

        let genome3 = vec![neuron(0, 2), input(0), bias(), neuron(1, 1), input(1)];
        let net3 = Network::new(genome3, Activation::Linear).unwrap();
        assert_eq!(2, net3.num_inputs());
        assert_eq!(2, net3.num_outputs());
        check_num_outputs(&net3);
    }

    #[test]
    fn test_neuron_info() {
        let (net, _, ()) =
            Network::load_file(get_file_path("test_network_multi_output.cge")).unwrap();

        let expected: HashMap<_, _> = [
            (NeuronId::new(0), NeuronInfo::new(0..5, 0)),
            (NeuronId::new(1), NeuronInfo::new(1..4, 1)),
            (NeuronId::new(2), NeuronInfo::new(5..9, 0)),
            (NeuronId::new(3), NeuronInfo::new(9..14, 0)),
            (NeuronId::new(4), NeuronInfo::new(11..14, 1)),
        ]
        .into_iter()
        .collect();
        assert_eq!(expected, net.neuron_info);
    }

    #[test]
    fn test_clear_state() {
        let (mut net, _, ()) =
            Network::load_file(get_file_path("test_network_recurrent.cge")).unwrap();

        let output = net.evaluate(&[]).unwrap().to_vec();
        let output2 = net.evaluate(&[]).unwrap().to_vec();

        assert_ne!(output, output2);

        net.clear_state();
        let output3 = net.evaluate(&[]).unwrap().to_vec();

        assert_eq!(output, output3);
    }

    #[test]
    fn test_next_neuron_id() {
        let genome = vec![neuron(0, 2), input(1), neuron(1, 1), bias()];
        let net = Network::new(genome, Activation::Linear).unwrap();

        assert_eq!(2, net.next_neuron_id);

        let genome2 = vec![neuron(2, 1), input(1)];
        let net2 = Network::new(genome2, Activation::Linear).unwrap();

        assert_eq!(3, net2.next_neuron_id);
    }

    #[test]
    fn test_validate_valid() {
        let genome = vec![neuron(0, 2), input(0), bias()];
        assert!(Network::new(genome, Activation::Linear).is_ok());

        let genome2 = vec![
            neuron(0, 5),
            input(0),
            bias(),
            forward(1),
            recurrent(1),
            neuron(1, 1),
            input(1),
        ];
        assert!(Network::new(genome2, Activation::Linear).is_ok());
    }

    #[test]
    fn test_validate_empty() {
        let genome = vec![];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::EmptyGenome
        );
    }

    #[test]
    fn test_validate_invalid_input_count() {
        let genome = vec![neuron(0, 1), neuron(2, 0), input(0)];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::InvalidInputCount(1, NeuronId::new(2))
        );
    }

    #[test]
    fn test_validate_not_enough_inputs() {
        let genome = vec![neuron(0, 2), neuron(2, 1), input(0)];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::NotEnoughInputs(0, NeuronId::new(0))
        );

        let genome2 = vec![neuron(1, 1)];
        assert_eq!(
            Network::new(genome2, Activation::Linear).unwrap_err(),
            Error::NotEnoughInputs(0, NeuronId::new(1))
        );

        let genome3 = vec![neuron(2, 3), bias(), input(0)];
        assert_eq!(
            Network::new(genome3, Activation::Linear).unwrap_err(),
            Error::NotEnoughInputs(0, NeuronId::new(2))
        );
    }

    #[test]
    fn test_validate_duplicate_neuron_id() {
        let genome = vec![neuron(1, 2), input(1), neuron(1, 1), bias()];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::DuplicateNeuronId(2, 0, NeuronId::new(1))
        );

        let genome2 = vec![
            neuron(0, 2),
            input(1),
            neuron(1, 2),
            bias(),
            neuron(1, 1),
            input(0),
        ];
        assert_eq!(
            Network::new(genome2, Activation::Linear).unwrap_err(),
            Error::DuplicateNeuronId(4, 2, NeuronId::new(1))
        );
    }

    #[test]
    fn test_validate_non_neuron_output() {
        for gene in [bias(), input(0), forward(1), recurrent(1)] {
            let genome = vec![gene];
            assert_eq!(
                Network::new(genome, Activation::Linear).unwrap_err(),
                Error::NonNeuronOutput(0)
            );
        }

        let genome2 = vec![neuron(0, 2), input(1), bias(), input(0)];
        assert_eq!(
            Network::new(genome2, Activation::Linear).unwrap_err(),
            Error::NonNeuronOutput(3)
        );

        let genome3 = vec![bias(), neuron(0, 1), input(0)];
        assert_eq!(
            Network::new(genome3, Activation::Linear).unwrap_err(),
            Error::NonNeuronOutput(0)
        );
    }

    #[test]
    fn test_validate_invalid_jumper_source() {
        let genome = vec![neuron(0, 1), forward(3), neuron(1, 1), bias()];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::InvalidJumperSource(1, NeuronId::new(3))
        );

        let genome2 = vec![neuron(0, 1), recurrent(2)];
        assert_eq!(
            Network::new(genome2, Activation::Linear).unwrap_err(),
            Error::InvalidJumperSource(1, NeuronId::new(2))
        );
    }

    #[test]
    fn test_validate_invalid_forward_jumper() {
        let genome = vec![neuron(0, 1), forward(0)];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::InvalidForwardJumper(1)
        );

        let genome2 = vec![
            neuron(0, 2),
            neuron(1, 1),
            input(0),
            neuron(2, 1),
            neuron(3, 1),
            forward(1),
        ];
        assert_eq!(
            Network::new(genome2, Activation::Linear).unwrap_err(),
            Error::InvalidForwardJumper(5)
        );
    }

    #[test]
    fn test_validate_extreme_neuron_input_count() {
        let genome = vec![neuron(usize::MAX, (usize::MAX / 2) + 1)];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::Arithmetic
        );
    }

    #[test]
    fn test_validate_extreme_neuron_id() {
        let genome = vec![neuron(usize::MAX, 1), bias()];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::Arithmetic
        );
    }

    #[test]
    fn test_validate_extreme_input_id() {
        let genome = vec![neuron(0, 1), input(usize::MAX)];
        assert_eq!(
            Network::new(genome, Activation::Linear).unwrap_err(),
            Error::Arithmetic
        );
    }
}
