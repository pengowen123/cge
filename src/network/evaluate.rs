//! Evaluation of networks.

use std::collections::HashMap;
use std::ops::{Index, Range};

use super::utils;
use crate::activation::Activation;
use crate::gene::{Gene, InputId, NeuronId};
use crate::network::NeuronInfo;
use crate::stack::Stack;

/// The inputs to a network.
#[derive(Clone, Copy)]
pub struct Inputs<'a>(pub &'a [f64]);

impl<'a> Index<InputId> for Inputs<'a> {
    type Output = f64;

    fn index(&self, index: InputId) -> &Self::Output {
        &self.0[index.as_usize()]
    }
}

/// Evaluates the subgenome in the given range. The output of the subgenome is placed on the stack.
///
/// If `ignore_final_neuron_weight` is `true`, the weight of the final neuron in the subgenome is
/// ignored.
pub fn evaluate_slice<'s>(
    genome: &mut Vec<Gene>,
    range: Range<usize>,
    inputs: Inputs,
    stack: &'s mut Stack,
    ignore_final_neuron_weight: bool,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    activation: Activation,
) {
    // Iterate backwards over the specified slice
    for (i, gene_index) in range.enumerate().rev() {
        let weight;
        let value;

        if genome[gene_index].is_bias() {
            // If it is a bias gene, push 1.0 and the gene's weight onto the stack
            let bias = genome[gene_index].as_bias().unwrap();
            weight = bias.value();
            value = 1.0;
        } else if genome[gene_index].is_input() {
            // If it is an input gene, push the corresponding input value and the gene's weight
            // onto the stack
            let input = genome[gene_index].as_input().unwrap();
            weight = input.weight();
            value = inputs[input.id()];
        } else if genome[gene_index].is_neuron() {
            let neuron = genome[gene_index].as_mut_neuron().unwrap();
            // If it is a neuron gene, pop the number of required inputs off the stack, and push
            // the sum of these inputs passed through the activation function and the gene's
            // weight onto the stack
            let sum_inputs = stack
                .pop_sum(neuron.num_inputs())
                .expect("A neuron did not receive enough inputs");

            // Apply the activation function
            value = activation.apply(sum_inputs);

            // Update the neuron's current value (unweighted)
            neuron.set_current_value(Some(value));

            if i == 0 && ignore_final_neuron_weight {
                // Ignore weight for the final neuron in the genome if the flag is set
                weight = 1.0;
            } else {
                weight = neuron.weight();
            }
        } else if genome[gene_index].is_forward_jumper() {
            // If it is a forward jumper gene, evaluate the subgenome of the source neuron and
            // push its output and the gene's weight onto the stack
            let forward = genome[gene_index].as_forward_jumper().unwrap();
            let source_subgenome_range = neuron_info[&forward.source_id()].subgenome_range();
            let source = genome[source_subgenome_range.start].as_neuron().unwrap();

            weight = forward.weight();

            let subgenome_output = if let Some(cached) = source.current_value() {
                cached
            } else {
                // NOTE: This is somewhat inefficient because it can run the neuron evaluation code
                //       up to two times (once in this subcall to evaluate_slice and once in the
                //       main evaluate_slice call) depending on the genome order
                //       Also, the call stack may grow in proportion to the genome length in the
                //       worst case (exactly reversed execution order of a chain of forward
                //       jumpers)
                //       Both of these could probably be fixed with a smart iteration solution
                //       instead of recursion, or if a graph structure is used to form a strict
                //       ordering of evaluation
                evaluate_slice(
                    genome,
                    source_subgenome_range,
                    inputs,
                    stack,
                    true,
                    neuron_info,
                    activation,
                );
                stack.pop().unwrap()
            };

            value = subgenome_output;
        } else if genome[gene_index].is_recurrent_jumper() {
            // If it is a recurrent jumper gene, push the previous value of the source neuron
            // and the gene's weight onto the stack
            let recurrent = genome[gene_index].as_recurrent_jumper().unwrap();
            let source = utils::get_neuron(recurrent.source_id(), neuron_info, genome).unwrap();

            weight = recurrent.weight();
            value = source.previous_value();
        } else {
            unreachable!();
        }

        // Push the weighted value onto the stack
        stack.push(weight * value);
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;
    use crate::network::NotEnoughInputsError;
    use crate::{Network, WithRecurrentState};

    fn get_file_path(file_name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), file_name)
    }

    #[test]
    fn test_evaluate_full() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_v1.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

        let output = net.evaluate(&[1.0, 1.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1.014);

        let output2 = net.evaluate(&[0.0, 0.0]).unwrap();
        assert_eq!(output2.len(), 1);
        assert_eq!(output2[0], 0.40056);
    }

    #[test]
    fn test_inputs() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_v1.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

        // Extra inputs should be discarded
        let output = net.evaluate(&[1.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1.014);

        let output2 = net.evaluate(&[0.0, 0.0, 2.0, 3.0]).unwrap();
        assert_eq!(output2.len(), 1);
        assert_eq!(output2[0], 0.40056);

        // Too few inputs returns `None`
        assert_eq!(Err(NotEnoughInputsError::new(2, 1)), net.evaluate(&[1.0]));
    }

    #[test]
    fn test_activation() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_v1.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

        // Check that the activation function is being applied
        net.set_activation(Activation::Tanh);

        let output = net.evaluate(&[1.0, 1.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_approx_eq!(output[0], 0.3913229613565932);

        let output2 = net.evaluate(&[0.0, 0.0]).unwrap();
        assert_eq!(output2.len(), 1);
        assert_approx_eq!(output2[0], 0.11798552468976746);
    }

    #[test]
    fn test_multiple_outputs() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_multi_output.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

        let inputs = [2.0, 3.0];
        let output = net.evaluate(&inputs).unwrap().to_vec();

        let expected = [3.541362029170628, 3.2752704637145316, 1.1087918551621792];

        assert_eq!(expected.len(), output.len());
        for i in 0..3 {
            assert_approx_eq!(expected[i], output[i]);
        }

        // There are no recurrent connections, so the output should remain constant
        let output2 = net.evaluate(&inputs).unwrap();
        assert_eq!(output, output2);
    }

    #[test]
    fn test_forward_jumper_cached() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_v1.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

        for gene in &mut net.genome {
            if let Gene::Neuron(neuron) = gene {
                // Insert dummy cached values
                neuron.set_current_value(Some(100.0));
            }
        }

        // Make sure they are used
        // They will be overwritten when each subnetwork is actually evaluated, but the only forward
        // jumper in the genome comes before its source subnetwork, so the dummy value will be used
        // before being overwritten
        let output = net.evaluate(&[0.0, 0.0]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 3.96);
    }

    #[test]
    fn test_recurrent_previous_value() {
        let (mut net, _, ()) = Network::load_file(
            get_file_path("test_network_recurrent.cge"),
            WithRecurrentState(false),
        )
        .unwrap();

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
