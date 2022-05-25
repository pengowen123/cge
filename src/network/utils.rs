//! Utilities for networks.

use std::collections::HashMap;

use super::NeuronInfo;
use crate::gene::{Gene, Neuron, NeuronId};

/// Returns a reference to the neuron with the given ID if it exists.
pub fn get_neuron<'a>(
    id: NeuronId,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    genome: &'a [Gene],
) -> Option<&'a Neuron> {
    neuron_info.get(&id).map(|info| {
        let source_index = info.subgenome_range().start;
        genome[source_index].as_neuron().unwrap()
    })
}

/// Returns a mutable reference to the neuron with the given ID if it exists.
pub fn get_mut_neuron<'a>(
    id: NeuronId,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    genome: &'a mut [Gene],
) -> Option<&'a mut Neuron> {
    neuron_info.get(&id).map(|info| {
        let source_index = info.subgenome_range().start;
        genome[source_index].as_mut_neuron().unwrap()
    })
}
