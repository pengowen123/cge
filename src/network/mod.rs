//! The neural network struct.

mod error;
mod evaluate;

pub use error::{Error, MutationError};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::iter;
use std::ops::{Index, Range};
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

// NOTE: All `Network` objects must be fully valid, and all methods assume this to be true
//       Any modifications to the genome must be matched with a validity check and corresponding
//       updates to the metadata
#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    // The genes of the network
    genome: Vec<Gene>,
    // The activation function to use for neuron outputs
    activation: Activation,
    // The ID to use for the next neuron added to the network
    next_neuron_id: usize,
    // Info about each neuron, updated when the genome is changed
    neuron_info: HashMap<NeuronId, NeuronInfo>,
    // Parent info for each gene
    gene_parents: Vec<Option<NeuronId>>,
    // The number of inputs required by the network (one plus the max input ID referred to)
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
            gene_parents: Vec::new(),
            num_inputs: 0,
            num_outputs: 0,
            stack: Stack::new(),
        };

        network.rebuild_metadata()?;

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
    fn rebuild_metadata(&mut self) -> Result<(), Error> {
        // O(n)

        // A stopping point to detect and handle the end of a previously-encountered subgenome
        struct StoppingPoint {
            // The counter value that signals the end of the subgenome
            counter: isize,
            // The ID of subgenome
            id: NeuronId,
            // The starting index of the subgenome
            start_index: usize,
            // The depth of the subgenome's root neuron
            depth: usize,
        }

        // Info needed to check the validity of a forward jumper
        struct ForwardJumperCheck {
            // The index of the forward jumper gene
            jumper_index: usize,
            // The depth of the forward jumper's parent neuron
            parent_depth: usize,
            // The source ID of the forward jumper
            source_id: NeuronId,
        }

        // Info needed to check the validity of a recurrent jumper
        struct RecurrentJumperCheck {
            // The index of the forward jumper gene
            jumper_index: usize,
            // The source ID of the forward jumper
            source_id: NeuronId,
        }

        if self.genome.is_empty() {
            return Err(Error::EmptyGenome);
        }

        let mut counter = 0isize;
        let neuron_info = &mut self.neuron_info;
        neuron_info.clear();
        let gene_parents = &mut self.gene_parents;
        gene_parents.clear();
        // Represents a stack of the current subgenomes being traversed
        // The value at the top of the stack when encountering a gene is that gene's parent
        let mut stopping_points: Vec<StoppingPoint> = Vec::new();
        // A list of info to check the validity of all forward jumpers after `neuron_info` is
        // completed
        let mut forward_jumper_checks: Vec<ForwardJumperCheck> = Vec::new();
        // A list of info to check the validity of all recurrent jumpers after `neuron_info` is
        // completed
        let mut recurrent_jumper_checks: Vec<RecurrentJumperCheck> = Vec::new();
        let mut max_input_id = None;
        let mut max_neuron_id = None;

        for (i, gene) in self.genome.iter().enumerate() {
            let parent = stopping_points.last().map(|p| p.id);
            let depth = stopping_points.len();
            // Each gene produces one output
            counter = counter.checked_add(1).ok_or(Error::Arithmetic)?;

            gene_parents.push(parent);

            if let Gene::Neuron(neuron) = gene {
                // Track the value of `counter` when encountering a new subgenome (neuron) so that
                // the end of the subgenome can be detected and handled
                // The subgenome's starting index and depth are also added
                stopping_points.push(StoppingPoint {
                    counter,
                    id: neuron.id(),
                    start_index: i,
                    depth,
                });

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
                if parent.is_none() {
                    return Err(Error::NonNeuronOutput(i));
                }

                // Add jumper info to be checked later
                match gene {
                    Gene::ForwardJumper(forward) => {
                        let parent_depth = depth.checked_sub(1).unwrap();
                        forward_jumper_checks.push(ForwardJumperCheck {
                            jumper_index: i,
                            parent_depth,
                            source_id: forward.source_id(),
                        });
                    }
                    Gene::RecurrentJumper(recurrent) => {
                        recurrent_jumper_checks.push(RecurrentJumperCheck {
                            jumper_index: i,
                            source_id: recurrent.source_id(),
                        });
                    }
                    Gene::Input(input) => {
                        max_input_id = max_input_id
                            .or(Some(0))
                            .map(|max_id| max_id.max(input.id().as_usize()));
                    }
                    _ => {}
                }

                // Check if `counter` has returned to its value from when any subgenomes started
                while !stopping_points.is_empty() && stopping_points.last().unwrap().counter == counter {
                    let stop = stopping_points.pop().unwrap();

                    if let Some(existing) = neuron_info.get(&stop.id) {
                        let existing_index = existing.subgenome_range().start;
                        return Err(Error::DuplicateNeuronId(existing_index, stop.start_index, stop.id));
                    }

                    let end_index = i.checked_add(1).unwrap();
                    let subgenome_range = stop.start_index..end_index;
                    neuron_info.insert(stop.id, NeuronInfo::new(subgenome_range, stop.depth));
                }
            }
        }

        // If any subgenomes were not fully traversed, a neuron did not receive enough inputs
        if let Some(stop) = stopping_points.last() {
            return Err(Error::NotEnoughInputs(stop.start_index, stop.id));
        }

        // Check that forward jumpers always connect parent neurons to source neurons of higher
        // depth
        for check in forward_jumper_checks {
            if let Some(source_info) = neuron_info.get(&check.source_id) {
                if check.parent_depth >= source_info.depth() {
                    return Err(Error::InvalidForwardJumper(check.jumper_index));
                }
            } else {
                // Return an error if the jumper's source does not exist
                return Err(Error::InvalidJumperSource(check.jumper_index, check.source_id));
            }
        }

        // Check that the source of every recurrent jumper exists
        for check in recurrent_jumper_checks {
            if !neuron_info.contains_key(&check.source_id) {
                return Err(Error::InvalidJumperSource(check.jumper_index, check.source_id));
            }
        }

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

    pub fn iter_neuron_ids<'a>(&'a self) -> impl Iterator<Item = NeuronId> + 'a {
        self.neuron_info.keys().cloned()
    }

    /// Returns the ID to be used for the next neuron added to this `Network`.
    pub fn next_neuron_id(&self) -> NeuronId {
        NeuronId::new(self.next_neuron_id)
    }

    /// Adds a non-neuron gene as an input to a `parent` [`Neuron`].
    pub fn add_non_neuron<G: Into<NonNeuronGene>>(
        &mut self,
        parent: NeuronId,
        gene: G,
    ) -> Result<(), MutationError> {
        self.add_genes(parent, None, vec![gene.into()]).map(|_| ())
    }

    /// Adds a sequence of non-neuron genes as an input to a `parent` [`Neuron`].
    pub fn add_non_neurons(
        &mut self,
        parent: NeuronId,
        genes: Vec<NonNeuronGene>,
    ) -> Result<(), MutationError> {
        self.add_genes(parent, None, genes).map(|_| ())
    }

    /// Adds a subnetwork (a [`Neuron`] gene with its inputs) as an input to a `parent` [`Neuron`].
    /// Returns the ID of the new subnetwork's neuron.
    ///
    /// The new neuron will have the ID given by [`next_neuron_id`][Self::next_neuron_id].
    /// Recurrent connections sourcing from the new neuron may be included in `inputs` by pointing
    /// them to this ID.
    pub fn add_subnetwork(
        &mut self,
        parent: NeuronId,
        weight: f64,
        inputs: Vec<NonNeuronGene>,
    ) -> Result<NeuronId, MutationError> {
        self.add_genes(parent, Some(weight), inputs)
            .map(Option::unwrap)
    }

    /// Adds a sequence of genes immediately following the `parent` neuron. Adds the genes as
    /// inputs to a new subnetwork if `subnetwork_weight` is `Some`. Checks that each gene is valid
    /// and updates any relevant network metadata.
    ///
    /// Returns the ID of the new subnetwork if added.
    fn add_genes(
        &mut self,
        parent: NeuronId,
        subnetwork_weight: Option<f64>,
        genes: Vec<NonNeuronGene>,
    ) -> Result<Option<NeuronId>, MutationError> {
        // O(n) on genes.len() + O(n) on the number of neurons in the genome
        if genes.is_empty() {
            return Err(MutationError::Empty);
        }

        let parent_info = self
            .neuron_info
            .get(&parent)
            .ok_or(MutationError::InvalidParent)?;
        let parent_index = parent_info.subgenome_range().start;

        // The index at which the new gene sequence starts
        let new_sequence_index = parent_index.checked_add(1).unwrap();

        // The ID of the new neuron if one is being added
        let new_neuron_id = subnetwork_weight.map(|_| NeuronId::new(self.next_neuron_id));
        // The parent of the genes in `genes`
        let parent_of_new_inputs = if let Some(id) = new_neuron_id {
            id
        } else {
            parent
        };
        // The depth of the new neuron if one is being added
        let new_neuron_depth = parent_info.depth().checked_add(1).unwrap();

        // The number of genes to be added to the genome
        let added_len = if new_neuron_id.is_some() {
            // One higher if a neuron is being added as well
            genes.len().checked_add(1).unwrap()
        } else {
            genes.len()
        };
        // The updated number of inputs to the `parent` neuron
        let parent_neuron = self[parent_index].as_neuron().unwrap();
        let new_parent_num_inputs = if new_neuron_id.is_some() {
            // The added subetwork is the only new input
            parent_neuron.num_inputs().checked_add(1).unwrap()
        } else {
            // Otherwise, all genes in `genes` are new inputs
            parent_neuron.num_inputs().checked_add(genes.len()).unwrap()
        };
        // Increment the next neuron ID if a new neuron was added
        let new_next_neuron_id = if let Some(id) = new_neuron_id {
            id.as_usize()
                .checked_add(1)
                .ok_or(MutationError::Arithmetic)?
        } else {
            self.next_neuron_id
        };
        // The new number of inputs to the network
        let mut new_num_inputs = self.num_inputs;

        // Validate the mutation
        {
            // No changes should be made until validation is fully completed to prevent partial
            // state updates
            let ref_self = &*self;

            for gene in &genes {
                match gene {
                    NonNeuronGene::Input(input) => {
                        // Update num_inputs
                        new_num_inputs = new_num_inputs.max(
                            input
                                .id()
                                .as_usize()
                                .checked_add(1)
                                .ok_or(MutationError::Arithmetic)?,
                        );
                    }
                    NonNeuronGene::ForwardJumper(forward) => {
                        // Check that any added forward jumpers point to higher depth neurons
                        let points_to_new_neuron = if let Some(id) = new_neuron_id {
                            forward.source_id() == id
                        } else {
                            false
                        };

                        if points_to_new_neuron {
                            return Err(MutationError::InvalidForwardJumper);
                        }

                        if let Some(info) = ref_self.neuron_info.get(&forward.source_id()) {
                            let mut parent_depth = ref_self[parent].depth();
                            // If adding a subnetwork, the parent is the subnetwork root, which has
                            // a depth of one higher than that of the `parent` argument given
                            if new_neuron_id.is_some() {
                                parent_depth = parent_depth.checked_add(1).unwrap();
                            }
                            if parent_depth >= info.depth() {
                                return Err(MutationError::InvalidForwardJumper);
                            }
                        } else {
                            return Err(MutationError::InvalidJumperSource);
                        }
                    }
                    NonNeuronGene::RecurrentJumper(recurrent) => {
                        // Check that any added recurrent jumpers point to neurons that exist or the
                        // new neuron if one is being added
                        let points_to_new_neuron = if let Some(id) = new_neuron_id {
                            recurrent.source_id() == id
                        } else {
                            false
                        };

                        if !(points_to_new_neuron
                            || ref_self.neuron_info.contains_key(&recurrent.source_id()))
                        {
                            return Err(MutationError::InvalidJumperSource);
                        }
                    }
                    NonNeuronGene::Bias(_) => {}
                }
            }
        }

        // All operations beyond this point must not return early in order to avoid leaving the
        // `Network` in a partially updated state

        // Update neuron info map
        for (_, info) in &mut self.neuron_info {
            if info.subgenome_range.start >= new_sequence_index {
                info.subgenome_range.start += added_len;
                info.subgenome_range.end += added_len;
            } else if info.subgenome_range.contains(&new_sequence_index) {
                info.subgenome_range.end += added_len;
            }
        }
        if let Some(id) = new_neuron_id {
            let new_info = NeuronInfo::new(
                new_sequence_index..new_sequence_index + added_len,
                new_neuron_depth,
            );
            self.neuron_info.insert(id, new_info);
        }

        // Update parent neuron inputs
        self.genome[parent_index]
            .as_mut_neuron()
            .unwrap()
            .set_num_inputs(new_parent_num_inputs);

        // Insert the genes
        let genes_len = genes.len();
        self.genome.splice(
            new_sequence_index..new_sequence_index,
            genes.into_iter().map(Into::into),
        );
        // Update gene parent info
        self.gene_parents.splice(
            new_sequence_index..new_sequence_index,
            iter::repeat(Some(parent_of_new_inputs)).take(genes_len),
        );
        // Insert the new neuron at the front of the sequence if one is being added
        if let Some(weight) = subnetwork_weight {
            let num_inputs = genes_len;
            self.genome.insert(
                new_sequence_index,
                Neuron::new(new_neuron_id.unwrap(), num_inputs, weight).into(),
            );
            self.gene_parents.insert(new_sequence_index, Some(parent));
        }

        // Update other metadata
        self.num_inputs = new_num_inputs;
        self.next_neuron_id = new_next_neuron_id;

        Ok(new_neuron_id)
    }

    /// Removes and returns the non-neuron gene at the index if it is not the only input to its
    /// parent neuron.
    pub fn remove_non_neuron(&mut self, index: usize) -> Result<Gene, MutationError> {
        // O(n) only to update_num_inputs, O(1) for everything else
        if let Some(removed_gene) = self.genome.get(index) {
            if removed_gene.is_neuron() {
                return Err(MutationError::RemoveNeuron);
            }

            let parent_id = self.gene_parents[index].unwrap();
            let parent_index = self[parent_id].subgenome_range().start;
            let parent = self.genome[parent_index].as_mut_neuron().unwrap();
            let num_inputs = parent.num_inputs();

            if num_inputs == 1 {
                return Err(MutationError::RemoveOnlyInput);
            }

            // Update parent neuron
            parent.set_num_inputs(num_inputs.checked_sub(1).unwrap());

            // Update metadata
            for (_, info) in &mut self.neuron_info {
                if info.subgenome_range.start > index {
                    // Decrement the ranges of all subgenomes following the removed gene
                    info.subgenome_range.start = info.subgenome_range.start.checked_sub(1).unwrap();
                    info.subgenome_range.end = info.subgenome_range.end.checked_sub(1).unwrap();
                } else if info.subgenome_range.contains(&index) {
                    // Shrink the ranges of all subgenomes containing the removed gene
                    info.subgenome_range.end = info.subgenome_range.end.checked_sub(1).unwrap();
                }
            }

            let mut new_max_input_id = None;
            for (i, gene) in self.genome.iter().enumerate() {
                // Check all inputs other than the removed gene to find the new number of inputs to
                // the network
                if let Gene::Input(input) = gene {
                    if i != index {
                        new_max_input_id = new_max_input_id
                            .or(Some(0))
                            .map(|max_id| max_id.max(input.id().as_usize()));
                    }
                }
            }
            self.num_inputs = new_max_input_id
                .map(|id| id.checked_add(1).unwrap())
                .unwrap_or(0);

            // Remove the gene
            self.gene_parents.remove(index);
            Ok(self.genome.remove(index))
        } else {
            Err(MutationError::RemoveInvalidIndex)
        }
    }

    /// Returns an iterator of indices of genes that are valid to remove with
    /// [`remove_non_neuron`][Self::remove_non_neuron].
    pub fn get_valid_removals<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        self.genome
            .iter()
            .zip(&self.gene_parents)
            .enumerate()
            .filter_map(|(i, (gene, parent))| {
                // Neurons can't be removed
                if gene.is_neuron() {
                    None
                } else {
                    let parent_index = self[parent.unwrap()].subgenome_range().start;
                    let num_inputs = self[parent_index].as_neuron().unwrap().num_inputs();

                    // Genes that are the sole input of their parent neuron can't be removed
                    if num_inputs > 1 {
                        Some(i)
                    } else {
                        None
                    }
                }
            })
    }

    /// Returns an iterator of neuron IDs with depths greater than `parent_depth`, which can be
    /// used as sources for a [`ForwardJumper`] gene under a parent neuron with depth `parent_depth`.
    pub fn get_valid_forward_jumper_sources<'a>(
        &'a self,
        parent_depth: usize,
    ) -> impl Iterator<Item = NeuronId> + 'a {
        self.neuron_info.iter().filter_map(move |(&id, info)| {
            if info.depth() > parent_depth {
                Some(id)
            } else {
                None
            }
        })
    }
}

impl Index<usize> for Network {
    type Output = Gene;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.genome[idx]
    }
}

impl Index<NeuronId> for Network {
    type Output = NeuronInfo;
    fn index(&self, idx: NeuronId) -> &Self::Output {
        &self.neuron_info[&idx]
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
    use rand::prelude::*;

    use super::*;
    use crate::encoding::Metadata;

    fn get_file_path(file_name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), file_name)
    }

    fn bias<G: From<Bias>>() -> G {
        Bias::new(1.0).into()
    }

    fn input<G: From<Input>>(id: usize) -> G {
        Input::new(InputId::new(id), 1.0).into()
    }

    fn neuron<G: From<Neuron>>(id: usize, num_inputs: usize) -> G {
        Neuron::new(NeuronId::new(id), num_inputs, 1.0).into()
    }

    fn forward<G: From<ForwardJumper>>(source_id: usize) -> G {
        ForwardJumper::new(NeuronId::new(source_id), 1.0).into()
    }

    fn recurrent<G: From<RecurrentJumper>>(source_id: usize) -> G {
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
        let genome = vec![neuron(0, 1), bias()];
        let net = Network::new(genome, Activation::Linear).unwrap();
        assert_eq!(0, net.num_inputs());
        assert_eq!(1, net.num_outputs());
        check_num_outputs(&net);

        let genome2 = vec![neuron(0, 2), input(0), bias()];
        let net2 = Network::new(genome2, Activation::Linear).unwrap();
        assert_eq!(1, net2.num_inputs());
        assert_eq!(1, net2.num_outputs());
        check_num_outputs(&net2);

        let genome3 = vec![neuron(0, 3), input(0), bias(), input(2)];
        let net3 = Network::new(genome3, Activation::Linear).unwrap();
        assert_eq!(3, net3.num_inputs());
        assert_eq!(1, net3.num_outputs());
        check_num_outputs(&net3);

        let genome4 = vec![neuron(0, 2), input(0), bias(), neuron(1, 1), input(1)];
        let net4 = Network::new(genome4, Activation::Linear).unwrap();
        assert_eq!(2, net4.num_inputs());
        assert_eq!(2, net4.num_outputs());
        check_num_outputs(&net4);
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

        let expected_parents = vec![
            None,
            Some(NeuronId::new(0)),
            Some(NeuronId::new(1)),
            Some(NeuronId::new(1)),
            Some(NeuronId::new(0)),
            None,
            Some(NeuronId::new(2)),
            Some(NeuronId::new(2)),
            Some(NeuronId::new(2)),
            None,
            Some(NeuronId::new(3)),
            Some(NeuronId::new(3)),
            Some(NeuronId::new(4)),
            Some(NeuronId::new(4)),
        ];
        assert_eq!(expected_parents, net.gene_parents);
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

    /// Creates a `Network` from the genome, runs the mutation on it, and checks that the internal
    /// state was not updated
    fn run_invalid_mutation_test<F: Fn(&mut Network) -> Result<(), MutationError>>(
        genome: Vec<Gene>,
        mutate: F,
        expected: MutationError,
    ) {
        let mut network = Network::new(genome, Activation::Linear).unwrap();
        let old = network.clone();

        assert_eq!(Err(expected), mutate(&mut network));
        assert_eq!(old, network);
    }

    #[test]
    fn test_mutate_invalid_parent() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| {
                let new_gene: NonNeuronGene = input(1);
                net.add_non_neuron(NeuronId::new(1), new_gene)
            },
            MutationError::InvalidParent,
        );
    }

    #[test]
    fn test_mutate_invalid_jumper_source() {
        let new_genes: [NonNeuronGene; 2] = [forward::<NonNeuronGene>(1), recurrent(1)];
        for new_gene in new_genes {
            run_invalid_mutation_test(
                vec![neuron(0, 1), input(0)],
                |net| net.add_non_neuron(NeuronId::new(0), new_gene.clone()),
                MutationError::InvalidJumperSource,
            );
        }
    }

    #[test]
    fn test_mutate_invalid_forward_jumper() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| {
                let new_gene: NonNeuronGene = forward(0);
                net.add_non_neuron(NeuronId::new(0), new_gene)
            },
            MutationError::InvalidForwardJumper,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 2), neuron(1, 1), input(1), input(0)],
            |net| {
                let new_gene: NonNeuronGene = forward(0);
                net.add_non_neuron(NeuronId::new(1), new_gene)
            },
            MutationError::InvalidForwardJumper,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| {
                let inputs = vec![forward(net.next_neuron_id().as_usize())];
                net.add_subnetwork(NeuronId::new(0), 1.0, inputs)
                    .map(|_| ())
            },
            MutationError::InvalidForwardJumper,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 1), neuron(1, 1), input(0)],
            |net| {
                // This mutation is valid when not adding a subnetwork, but under a subnetwork the
                // depth is one higher, making it invalid
                let inputs = vec![forward(1)];
                net.add_subnetwork(NeuronId::new(0), 1.0, inputs)
                    .map(|_| ())
            },
            MutationError::InvalidForwardJumper,
        );
    }

    #[test]
    fn test_mutate_empty() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| {
                net.add_subnetwork(NeuronId::new(0), 1.0, vec![])
                    .map(|_| ())
            },
            MutationError::Empty,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| net.add_non_neurons(NeuronId::new(0), vec![]),
            MutationError::Empty,
        );
    }

    #[test]
    fn test_mutate_remove_invalid_index() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| net.remove_non_neuron(2).map(|_| ()),
            MutationError::RemoveInvalidIndex,
        );
    }

    #[test]
    fn test_mutate_remove_neuron() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), neuron(1, 1), input(0)],
            |net| net.remove_non_neuron(1).map(|_| ()),
            MutationError::RemoveNeuron,
        );
    }

    #[test]
    fn test_mutate_remove_only_input() {
        run_invalid_mutation_test(
            vec![neuron(0, 1), bias()],
            |net| net.remove_non_neuron(1).map(|_| ()),
            MutationError::RemoveOnlyInput,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 2), neuron(1, 1), input(0), bias()],
            |net| net.remove_non_neuron(2).map(|_| ()),
            MutationError::RemoveOnlyInput,
        );
    }

    #[test]
    fn test_mutate_arithmetic() {
        run_invalid_mutation_test(
            vec![neuron(usize::MAX - 1, 1), input(0)],
            |net| {
                let inputs = vec![bias()];
                net.add_subnetwork(NeuronId::new(usize::MAX - 1), 1.0, inputs)
                    .map(|_| ())
            },
            MutationError::Arithmetic,
        );

        run_invalid_mutation_test(
            vec![neuron(0, 1), input(0)],
            |net| {
                let new_gene: NonNeuronGene = input(usize::MAX);
                net.add_non_neuron(NeuronId::new(0), new_gene).map(|_| ())
            },
            MutationError::Arithmetic,
        );
    }

    /// Checks that the networks metadata does not change if a full rebuild is performed
    fn check_mutated_metadata(net: &mut Network) {
        let mutated_neuron_info = net.neuron_info.clone();
        let mutated_gene_parents = net.gene_parents.clone();
        let mutated_num_inputs = net.num_inputs();
        let num_outputs = net.num_outputs;
        let mutated_next_neuron_id = net.next_neuron_id();

        assert_eq!(Ok(()), net.rebuild_metadata());

        assert_eq!(net.neuron_info, mutated_neuron_info);
        assert_eq!(net.gene_parents, mutated_gene_parents);
        assert_eq!(net.num_inputs(), mutated_num_inputs);
        assert_eq!(num_outputs, net.num_outputs());
        assert_eq!(net.next_neuron_id(), mutated_next_neuron_id);
    }

    /// Creates a network from the genome, runs the mutation on it, and check that its genome and
    /// metadata are correct
    fn run_mutation_test<O, F: Fn(&mut Network) -> O>(
        start_genome: Vec<Gene>,
        mutate: F,
        end_genome: Vec<Gene>,
        expected_num_inputs: usize,
        expected_next_neuron_id: NeuronId,
    ) {
        let mut network = Network::new(start_genome, Activation::Linear).unwrap();
        let old_num_outputs = network.num_outputs();

        let _ = mutate(&mut network);

        assert_eq!(end_genome, network.genome());
        assert_eq!(expected_num_inputs, network.num_inputs());
        assert_eq!(old_num_outputs, network.num_outputs());
        assert_eq!(expected_next_neuron_id, network.next_neuron_id());

        // Check that evaluation works and doesn't crash
        assert!(network.evaluate(&[1.0; 10]).is_some());

        // Check that the metadata is mutated in a way that is equivalent to rebuilding it
        check_mutated_metadata(&mut network);
    }

    #[test]
    fn test_add_non_neuron() {
        run_mutation_test(
            vec![neuron(0, 1), neuron(1, 1), input(0)],
            |net| {
                let new_gene: NonNeuronGene = input(1);
                net.add_non_neuron(NeuronId::new(0), new_gene).unwrap();
            },
            vec![neuron(0, 2), input(1), neuron(1, 1), input(0)],
            2,
            NeuronId::new(2),
        );
    }

    #[test]
    fn test_add_non_neurons() {
        run_mutation_test(
            vec![neuron(0, 1), neuron(1, 1), input(0)],
            |net| {
                // Add several genes
                let new_genes = vec![bias(), input(0), forward(1), recurrent(0)];
                net.add_non_neurons(NeuronId::new(0), new_genes).unwrap();
            },
            vec![
                neuron(0, 5),
                bias(),
                input(0),
                forward(1),
                recurrent(0),
                neuron(1, 1),
                input(0),
            ],
            1,
            NeuronId::new(2),
        );
    }

    #[test]
    fn test_add_subnetwork() {
        run_mutation_test(
            vec![neuron(0, 1), neuron(1, 1), neuron(2, 1), input(0)],
            |net| {
                // Add several genes
                let new_genes = vec![
                    bias(),
                    input(1),
                    forward(2),
                    // This connection points to the to-be-added subnetwork
                    recurrent(net.next_neuron_id().as_usize()),
                ];
                net.add_subnetwork(NeuronId::new(0), 1.0, new_genes)
                    .unwrap();
            },
            vec![
                neuron(0, 2),
                neuron(3, 4),
                bias(),
                input(1),
                forward(2),
                recurrent(3),
                neuron(1, 1),
                neuron(2, 1),
                input(0),
            ],
            2,
            NeuronId::new(4),
        );

        run_mutation_test(
            vec![
                neuron(0, 1),
                neuron(1, 1),
                neuron(2, 1),
                neuron(3, 1),
                input(0),
            ],
            |net| {
                // Add several genes
                let new_genes = vec![
                    bias(),
                    input(1),
                    forward(3),
                    recurrent(net.next_neuron_id().as_usize()),
                ];
                net.add_subnetwork(NeuronId::new(0), 1.0, new_genes)
                    .unwrap();
            },
            vec![
                neuron(0, 2),
                neuron(4, 4),
                bias(),
                input(1),
                forward(3),
                recurrent(4),
                neuron(1, 1),
                neuron(2, 1),
                neuron(3, 1),
                input(0),
            ],
            2,
            NeuronId::new(5),
        );
    }

    #[test]
    fn test_remove_non_neuron() {
        run_mutation_test(
            vec![neuron(0, 2), input(3), input(0)],
            |net| assert_eq!(input::<Gene>(3), net.remove_non_neuron(1).unwrap()),
            vec![neuron(0, 1), input(0)],
            1,
            NeuronId::new(1),
        );

        run_mutation_test(
            vec![neuron(0, 1), neuron(1, 2), input(1), input(0)],
            |net| assert_eq!(input::<Gene>(0), net.remove_non_neuron(3).unwrap()),
            vec![neuron(0, 1), neuron(1, 1), input(1)],
            2,
            NeuronId::new(2),
        );

        run_mutation_test(
            vec![neuron(0, 2), neuron(1, 1), input(3), input(0)],
            |net| assert_eq!(input::<Gene>(0), net.remove_non_neuron(3).unwrap()),
            vec![neuron(0, 1), neuron(1, 1), input(3)],
            4,
            NeuronId::new(2),
        );

        run_mutation_test(
            vec![neuron(0, 3), input(0), neuron(1, 1), bias(), bias()],
            |net| assert_eq!(input::<Gene>(0), net.remove_non_neuron(1).unwrap()),
            vec![neuron(0, 2), neuron(1, 1), bias(), bias()],
            0,
            NeuronId::new(2),
        );
    }

    #[test]
    fn test_multiple_mutations() {
        // Assemble a complex network from a simple one through mutations
        run_mutation_test(
            vec![neuron(0, 1), bias()],
            |net| {
                let id = |g: NonNeuronGene| g;
                net.add_non_neuron(NeuronId::new(0), id(input(0))).unwrap();
                net.add_non_neuron(NeuronId::new(0), id(input(1))).unwrap();

                let subnetwork_1 = net
                    .add_subnetwork(NeuronId::new(0), 1.0, vec![recurrent(0), bias()])
                    .unwrap();

                net.add_non_neuron(subnetwork_1, id(input(0))).unwrap();

                let subnetwork_2 = net
                    .add_subnetwork(subnetwork_1, 1.0, vec![input(0), input(2)])
                    .unwrap();

                let index = net
                    .genome()
                    .iter()
                    .enumerate()
                    .find(|(_, gene)| {
                        if let Gene::Input(input) = gene {
                            input.id() == InputId::new(2)
                        } else {
                            false
                        }
                    })
                    .unwrap()
                    .0;
                net.remove_non_neuron(index).unwrap();

                net.add_non_neuron(subnetwork_2, id(recurrent(subnetwork_1.as_usize())))
                    .unwrap();
                net.add_non_neuron(subnetwork_1, id(bias())).unwrap();
                net.add_non_neuron(NeuronId::new(0), id(forward(subnetwork_1.as_usize())))
                    .unwrap();
                net.add_non_neuron(NeuronId::new(0), id(forward(subnetwork_2.as_usize())))
                    .unwrap();
            },
            vec![
                neuron(0, 6),
                forward(2),
                forward(1),
                neuron(1, 5),
                bias(),
                neuron(2, 2),
                recurrent(1),
                input(0),
                input(0),
                recurrent(0),
                bias(),
                input(1),
                input(0),
                bias(),
            ],
            2,
            NeuronId::new(3),
        );
    }

    #[test]
    fn test_get_valid_removals() {
        let genome = vec![
            neuron(0, 3),
            input(0),
            neuron(1, 1),
            neuron(2, 2),
            input(1),
            neuron(3, 1),
            bias(),
            forward(2),
        ];
        let mut net = Network::new(genome, Activation::Linear).unwrap();

        assert_eq!(&[1, 4, 7][..], net.get_valid_removals().collect::<Vec<_>>());

        // Check that each index actually represents a valid removal
        loop {
            let removals = net.get_valid_removals().collect::<Vec<_>>();

            if removals.is_empty() {
                check_mutated_metadata(&mut net);
                break;
            }

            net.remove_non_neuron(removals[0]).unwrap();

            let _ = net.evaluate(&[2.0, 3.0]);
        }
    }

    #[test]
    fn test_get_valid_forward_jumper_sources() {
        let genome = vec![
            neuron(0, 2),
            neuron(1, 1),
            bias(),
            neuron(2, 2),
            neuron(3, 1),
            neuron(4, 1),
            bias(),
            neuron(5, 1),
            bias(),
        ];
        let mut net = Network::new(genome, Activation::Linear).unwrap();
        let parent_id = NeuronId::new(1);
        let parent_depth = net[parent_id].depth();
        let valid_sources = net
            .get_valid_forward_jumper_sources(parent_depth)
            .collect::<Vec<_>>();

        for id in [NeuronId::new(3), NeuronId::new(4), NeuronId::new(5)] {
            assert!(valid_sources.contains(&id));
        }
        assert_eq!(3, valid_sources.len());

        // Check that each source is actually valid
        for id in &valid_sources {
            let forward: NonNeuronGene = forward(id.as_usize());
            assert!(net.add_non_neuron(parent_id, forward).is_ok());
        }

        // Check that each source not listed is actually invalid
        for id in 0..net.neuron_info.len() {
            if !valid_sources.contains(&NeuronId::new(id)) {
                let forward: NonNeuronGene = forward(id);
                assert!(net.add_non_neuron(parent_id, forward).is_err());
            }
        }
    }

    /// Returns a random `NonNeuronGene`
    fn get_random_non_neuron(net: &mut Network) -> NonNeuronGene {
        let mut rng = rand::thread_rng();

        let mut ids = net.iter_neuron_ids().collect::<Vec<_>>();

        // Add the neuron ID of the subnetwork being added (will be invalid if none is
        // being added)
        ids.push(net.next_neuron_id());

        match rng.gen_range(0..=3) {
            0 => bias(),
            1 => input(rng.gen_range(0..10)),
            2 => {
                let source = ids.choose(&mut rng).unwrap();
                forward(source.as_usize())
            }
            3 => {
                let source = ids.choose(&mut rng).unwrap();
                recurrent(source.as_usize())
            }
            _ => unreachable!(),
        }
    }

    /// Tries to add a random non-neuron gene to the network
    fn add_random_non_neuron(net: &mut Network, parent: NeuronId) {
        let new_gene = get_random_non_neuron(net);
        let _result = net.add_non_neuron(parent, new_gene);
    }

    /// Tries to add a random sequence of non-neuron genes to the network
    fn add_random_non_neurons(net: &mut Network, parent: NeuronId) {
        let mut rng = rand::thread_rng();
        let count = rng.gen_range(0..=2);
        let new_genes = (0..count).map(|_| get_random_non_neuron(net)).collect();

        let _result = net.add_non_neurons(parent, new_genes);
    }

    /// Tries to add a random subnetwork to the network
    fn add_random_subnetwork(net: &mut Network, parent: NeuronId) {
        let mut rng = rand::thread_rng();
        let num_inputs = rng.gen_range(0..=3);
        let inputs = (0..num_inputs)
            .map(|_| get_random_non_neuron(net))
            .collect();

        let _result = net.add_subnetwork(parent, 1.0, inputs);
    }

    /// Tries to remove a random gene from the network
    fn remove_random_gene(net: &mut Network) {
        let mut rng = rand::thread_rng();
        let index = (0..=net.genome().len()).choose(&mut rng).unwrap();

        let _result = net.remove_non_neuron(index);
    }

    /// Builds and tests a random network from the initial genome using mutation operators
    /// Attempts invalid mutations in addition to valid ones, with a somewhat even split between
    /// them
    fn build_random_network(initial: Vec<Gene>) {
        const MUTATION_COUNT: usize = 200;

        let mut network = Network::new(initial, Activation::Linear).unwrap();
        let initial_outputs = network.num_outputs();
        let mut rng = rand::thread_rng();

        for _ in 0..MUTATION_COUNT {
            let parent = (0..=network.next_neuron_id().as_usize())
                .choose(&mut rng)
                .unwrap();
            let parent = NeuronId::new(parent);

            match rng.gen_range(0..=3) {
                0 => add_random_non_neuron(&mut network, parent),
                1 => add_random_non_neurons(&mut network, parent),
                2 => add_random_subnetwork(&mut network, parent),
                3 => remove_random_gene(&mut network),
                _ => unreachable!(),
            }

            check_mutated_metadata(&mut network);

            assert!(network.evaluate(&[1.0; 10]).is_some());
            network.clear_state();
        }

        assert_eq!(initial_outputs, network.num_outputs());

        let string = network.to_string(Metadata::new(None), ()).unwrap();
        let (converted_network, _, ()) = Network::load_str(&string).unwrap();

        network.stack.clear();
        network.clear_state();
        assert_eq!(converted_network, network);
    }

    #[test]
    fn test_random() {
        for _ in 0..10 {
            build_random_network(vec![neuron(0, 1), input(0)]);
            build_random_network(vec![neuron(0, 1), input(0), neuron(1, 1), input(1)]);
            build_random_network(vec![
                neuron(0, 1),
                input(0),
                neuron(1, 1),
                input(1),
                neuron(2, 1),
                input(2),
            ]);
        }
    }
}
