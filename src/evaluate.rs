//! Evaluation of networks.

use std::ops::{Index, Range};
use std::collections::HashMap;

use crate::gene::{Gene, NeuronId, InputId};
use crate::activation::Activation;
use crate::network::NeuronInfo;
use crate::utils::Stack;

/// The inputs to a network.
#[derive(Clone, Copy)]
pub struct Inputs<'a>(pub &'a [f64]);

impl<'a> Index<InputId> for Inputs<'a> {
    type Output = f64;

    fn index(&self, index: InputId) -> &Self::Output {
        &self.0[index.as_usize()]
    }
}

/// Returns the output of the subgenome in the given range.
///
/// If `ignore_final_neuron_weight` is `true`, the weight of the final neuron in the subgenome is
/// ignored.
pub fn evaluate_slice(
    genome: &mut Vec<Gene>,
    range: Range<usize>,
    inputs: Inputs,
    ignore_final_neuron_weight: bool,
    neuron_info: &HashMap<NeuronId, NeuronInfo>,
    activation: Activation,
) -> Vec<f64> {
    // Initialize a stack for evaluating the neural network
    let mut stack = Stack::new();

    // Iterate backwards over the specified slice
    for (i, gene_index) in range.enumerate().rev() {
        let weight;
        let value;

        if genome[gene_index].is_input() {
            if let Gene::Input(input) = &genome[gene_index] {
                // If it is an input gene, push the corresponding input value and the gene's weight
                // onto the stack
                weight = input.weight();
                value = inputs[input.id()];
            } else {
                unreachable!();
            }
        } else if genome[gene_index].is_neuron() {
            if let Gene::Neuron(neuron) = &mut genome[gene_index] {
                // If it is a neuron gene, pop the number of required inputs off the stack, and push
                // the sum of these inputs passed through the activation function and the gene's
                // weight onto the stack
                let sum_inputs = stack
                    .pop(neuron.num_inputs())
                    .expect("A neuron did not receive enough inputs")
                    .iter()
                    .sum();

                // Apply the activation function
                value = activation.get_func()(sum_inputs);

                // Update the neuron's current value (unweighted)
                neuron.set_current_value(Some(value));

                if i == 0 && ignore_final_neuron_weight {
                    // Ignore weight for the final neuron in the genome if the flag is set
                    weight = 1.0;
                } else {
                    weight = neuron.weight();
                }
            } else {
                unreachable!();
            }
        } else if genome[gene_index].is_forward_jumper() {
            // If it is a forward jumper gene, evaluate the subgenome of the source neuron and
            // push its output and the gene's weight onto the stack
            let source_subgenome_range;

            if let Gene::ForwardJumper(forward) = &genome[gene_index] {
                source_subgenome_range = neuron_info[&forward.source_id()].subgenome_range();
                weight = forward.weight();
            } else {
                unreachable!();
            }

            let subgenome_root = match &genome[source_subgenome_range.start] {
                Gene::Neuron(neuron) => neuron,
                _ => panic!("forward jumper source is not a neuron"),
            };

            let subgenome_output = if let Some(cached) = subgenome_root.current_value() {
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
                    true,
                    neuron_info,
                    activation,
                )[0]
            };

            value = subgenome_output;
        } else if genome[gene_index].is_recurrent_jumper() {
            if let Gene::RecurrentJumper(recurrent) = &genome[gene_index] {
                // If it is a recurrent jumper gene, push the previous value of the source neuron
                // and the gene's weight onto the stack
                let index = neuron_info[&recurrent.source_id()].subgenome_range().start;
                let source_gene = &genome[index];

                weight = recurrent.weight();
                if let Gene::Neuron(neuron) = source_gene {
                    value = neuron.previous_value();
                } else {
                    panic!("recurrent jumper did not point to a neuron");
                }
            } else {
                unreachable!();
            }
        } else if genome[gene_index].is_bias() {
            if let Gene::Bias(bias) = &genome[gene_index] {
                // If it is a bias gene, push 1.0 and the gene's weight onto the stack
                weight = bias.value();
                value = 1.0;
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }

        // Push the weighted value onto the stack
        stack.push(weight * value);
    }

    stack.data
}
