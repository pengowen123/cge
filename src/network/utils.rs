//! Utilities for networks.

use num_traits::Float;
use std::collections::HashMap;

use super::NeuronInfo;
use crate::gene::{Gene, Neuron, NeuronId};

/// Returns a reference to the neuron with the given ID if it exists.
pub fn get_neuron<'a, T: Float>(
    id: NeuronId,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    genome: &'a [Gene<T>],
) -> Option<&'a Neuron<T>> {
    if let Some(info) = neuron_info.get(&id) {
        let source_index = info.subgenome_range().start;
        Some(genome[source_index].as_neuron().unwrap())
    } else {
        None
    }
}

/// Returns a mutable reference to the neuron with the given ID if it exists.
pub fn get_mut_neuron<'a, T: Float>(
    id: NeuronId,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    genome: &'a mut [Gene<T>],
) -> Option<&'a mut Neuron<T>> {
    if let Some(info) = neuron_info.get(&id) {
        let source_index = info.subgenome_range().start;
        Some(genome[source_index].as_mut_neuron().unwrap())
    } else {
        None
    }
}
