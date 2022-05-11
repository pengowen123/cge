//! The neural network struct.

use std::ops::Range;
use std::io;

use crate::utils::Stack;
use crate::file;
use crate::gene::*;
use crate::gene::GeneExtras::*;
use crate::activation::*;

const BIAS_GENE_VALUE: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    // size should be the length of the genome minus one, don't forget
    pub size: usize,
    pub genome: Vec<Gene>,
    pub function: Activation
}

impl Network {
    /// Evaluates the neural network with the given inputs, returning a vector of outputs. The encoding can
    /// encode recurrent connections and bias inputs, so an internal state is used. It is important to run
    /// the clear_state method before calling evaluate again, unless it is desired to allow data
    /// carry over from the previous evaluation, for example if the network is being used as a real
    /// time controller.
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
    /// let result = network.evaluate(&vec![1.0, 1.0]);
    ///
    /// // Get the output of the neural network with no inputs
    /// let result = network.evaluate(&[]);
    ///
    /// // Get the output of the neural network with too many inputs (extras aren't used)
    /// let result = network.evaluate(&[1.0, 1.0, 1.0]);
    /// 
    /// // Let's say adder.ann is a file with a neural network with recurrent connections, used for
    /// // adding numbers together.
    /// let mut adder = Network::load_from_file("adder.ann").unwrap();
    ///
    /// // result_one will be 1.0
    /// let result_one = adder.evaluate(&[1.0]);
    ///
    /// // result_two will be 3.0
    /// let result_two = adder.evaluate(&[2.0]);
    ///
    /// // result_three will be 5.0
    /// let result_three = adder.evaluate(&[2.0]);
    ///
    /// // If this behavior is not desired, call the clear_state method between evaluations:
    /// let result_one = adder.evaluate(&[1.0]);
    ///
    /// adder.clear_state();
    ///
    /// // The 1.0 from the previous call is gone, so result_two will be 2.0
    /// let result_two = adder.evaluate(&[2.0]);
    /// ```
    pub fn evaluate(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.set_inputs(inputs);

        let size = self.size;
        let result = self.evaluate_slice(0..size, true, false);

        self.update_previous_values();

        result
    }

    /// Clears the internal state of the neural network.
    pub fn clear_state(&mut self) {
        for gene in &mut self.genome {
            match gene.variant {
                Input(ref mut current_value) => {
                    *current_value = 0.0;
                },
                Neuron(ref mut current_value, ref mut previous_value, _) => {
                    *current_value = 0.0;
                    *previous_value = 0.0;
                }
                _ => {}
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
    ///
    /// # Examples
    ///
    /// ```
    /// use cge::*;
    ///
    /// // Create a neural network
    /// let network = Network {
    ///     size: 0,
    ///     genome: Vec::new(),
    ///     function: Activation::Sign
    /// };
    ///
    /// // Save the neural network to the string
    /// let string = network.to_str();
    /// ```
    pub fn to_str(&self) -> String {
        file::to_str(self)
    }

    /// Saves the neural network to a file. Returns an empty tuple on success, or an io error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cge::*;
    ///
    /// // Create a neural network
    /// let network = Network {
    ///     size: 0,
    ///     genome: Vec::new(),
    ///     function: Activation::Sign
    /// };
    ///
    /// // Save the neural network to neural_network.ann
    /// network.save_to_file("neural_network.ann").unwrap();
    /// ```
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        file::write_network(self, path)
    }

    /// Loads a neural network from a file. No guarantees are made about the validity of the
    /// genome. Returns the network, or an io error. If the file is in a bad format,
    /// `std::io::ErrorKind::InvalidData` is returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cge::Network;
    ///
    /// // Loads a neural network from the file neural_network.ann
    /// let mut network = Network::load_from_file("neural_network.ann").unwrap();
    /// ```
    pub fn load_from_file(path: &str) -> io::Result<Network> {
        file::read_network(path)
    }

    // Returns the output of sub-linear genome in the given range
    // The j flag indicates whether or not the last node in the jump forward connection calculation is being evaluated,
    // in that case, do not include regular connection weight of neuron as this would be incorrect
    fn evaluate_slice(&mut self, range: Range<usize>, neuron_update: bool, j: bool) -> Vec<f64> {
        debug!("evaluate_slice in range: {:?}", range);

        let mut gene_index = range.end;
        // Initialize a stack for evaluating the neural network
        let mut stack = Stack::new();

        // TODO: activation function for each node
        let act_func = self.function.get_func();

        // Iterate backwards over the specified slice
        while gene_index >= range.start {
            let variant = self.genome[gene_index].variant;

            match variant {
                Input(_) => {
                    // If the gene is an input, push its value multiplied by the inputs weight onto
                    // the stack
                    let (weight, _, value) = self.genome[gene_index].ref_input().unwrap();
                    stack.push(weight * value);
                },
                Neuron(_, _, _) => {
                    // If the gene is a neuron, pop a number (the neurons input count) of inputs
                    // off the stack, and push the transfer function applied to the sum of these
                    // inputs multiplied by the neurons weight onto the stack
                    let (weight, _, current_value, _, inputs)
                        = self.genome[gene_index].ref_mut_neuron().unwrap();
                    let mut new_value = stack.pop(*inputs)
                        .expect("A neuron did not receive enough inputs")
                        .iter()
                        .fold(0.0, |acc, i| acc + i);

                    // apply the activation function
                    new_value = (act_func)(new_value);

                    // Store the neuron's current value in order to update the previous value later
                    if neuron_update {
                        *current_value = new_value;
                    }

                    if j && gene_index == range.start {
                        // when j flag is set,
                        // do not include weight of last neuron link as jump forward has a different weight
                        stack.push(new_value);
                    } else {
                        // otherwise use regular weight of connection in stack
                        stack.push(*weight * new_value);
                    }
                },
                Forward => {
                    // This is inefficient because it can run the neuron evaluation code multiple
                    // times
                    // TODO: Turn current value of neurons into a struct with a flag representing
                    // whether the neuron has been evaluated this network evaluation. Reset this
                    // flag after every network evaluation.

                    // If the gene is a forward jumper, evaluate the subnetwork starting at the
                    // neuron with id of the jumper, and push the result multiplied by the jumpers
                    // weight onto the stack
                    let weight = self.genome[gene_index].weight;
                    let id = self.genome[gene_index].id;
                    let subnetwork_range = self.get_subnetwork_index(id)
                        .expect("Found forward connection with invalid neuron id");

                    // set j flag to true so the neuron does not include it's regular link weight
                    // otherwise the values will be off by whatever factor the neuron weight is
                    let result = self.evaluate_slice(subnetwork_range, false, true);

                    debug!("{:?}", result);

                    stack.push(weight * result[0]);
                },
                Recurrent => {
                    // If the gene is a recurrent jumper, push the previous value of the neuron
                    // with the id of the jumper multiplied by the jumpers weight onto the stack
                    let gene = &self.genome[gene_index];
                    let neuron = &self.genome[self.get_neuron_index(gene.id)
                        .expect("Found recurrent connection with invalid neuron id")];

                    if let Neuron(_, previous_value, _) = neuron.variant {
                        stack.push(gene.weight * previous_value);
                    }
                },
                Bias => {
                    // If the gene is a bias input, push the bias constant multiplied by the genes
                    // weight onto the stack
                    let gene = &self.genome[gene_index];
                    stack.push(gene.weight * BIAS_GENE_VALUE);
                }
            }

            if gene_index == range.start {
                break;
            }

            gene_index -= 1;

            debug!("{:?}", stack.data);
        }

        stack.data
    }

    fn update_previous_values(&mut self) {
        for gene in &mut self.genome {
            if let Neuron(ref current_value, ref mut previous_value, _) = gene.variant {
                *previous_value = *current_value;
            }
        }
    }

    fn set_inputs(&mut self, inputs: &[f64]) {
        for gene in &mut self.genome {
            if let Input(ref mut current_value) = gene.variant {
                *current_value = 0.0;

                *current_value = match inputs.get(gene.id) {
                    Some(v) => *v,
                    None => 0.0
                }
            }
        }
    }

    /// Returns the start and end index of the subnetwork starting at the neuron with the given id,
    /// or None if it does not exist.
    pub fn get_subnetwork_index(&self, id: usize) -> Option<Range<usize>> {
        let start = match self.get_neuron_index(id) {
            Some(i) => i,
            None => return None
        };

        let mut end = start;
        let mut sum = 0;

        // Iterate through genes after the start index, modifying the sum each step
        // I could use an iterator here, but it would be messy
        for gene in &self.genome[start..self.size + 1] {
            match gene.variant {
                Neuron(_, _, ref inputs) => {
                    sum += 1 - *inputs as i32;
                },
                _ => {
                    sum += 1;
                }
            }

            if sum == 1 {
                break;
            }

            end += 1;
        }

        if sum != 1 {
            None
        } else {
            Some(Range {
                start,
                end
            })
        }
    }

    /// Returns the index of the neuron with the given id, or None if it does not exist.
    pub fn get_neuron_index(&self, id: usize) -> Option<usize> {
        let mut result = None;

        for (i, gene) in self.genome.iter().enumerate() {
            if let Neuron(_, _, _) = gene.variant {
                if gene.id == id {
                    result = Some(i);
                }
            }
        }

        result
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // linear genome from fig. 5.3 in paper:
    // https://www.researchgate.net/profile/Yohannes_Kassahun/publication/266864021_Towards_a_Unified_Approach_to_Learning_and_Adaptation/links/54ba91790cf253b50e2d037d.pdf?origin=publication_detail
    pub(crate) const TEST_GENOME: [Gene; 11] = [
        Gene {
            weight: 0.6,
            id: 0,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.8,
            id: 1,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.9,
            id: 3,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.1,
            id: 0,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.4,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.5,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.2,
            id: 2,
            variant: Neuron(0.0, 0.0, 4)
        },
        Gene {
            weight: 0.3,
            id: 3,
            variant: Forward
        },
        Gene {
            weight: 0.7,
            id: 0,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.8,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.2,
            id: 0,
            variant: Recurrent
        },
    ];

    // This genome has one more neuron than TEST_GENOME
    // It is placed between neuron 3 and input id 1 by splitting one connection into two
    // Therefore the Genome has one more gene
    // The link weight connecting neuron 3 and 4 is 0.2,
    // The link weight connecting neuron 4 and input 1 is 0.3
    // removed original connection gene from neuron 3 to input 1
    const TEST_GENOME_2: [Gene; 12] = [
        Gene {
            weight: 0.6,
            id: 0,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.8,
            id: 1,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.9,
            id: 3,
            variant: Neuron(0.0, 0.0, 2)
        },
        Gene {
            weight: 0.1,
            id: 0,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.2,
            id: 4,
            variant: Neuron(0.0, 0.0, 1)
        },
        Gene {
            weight: 0.3,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.5,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.2,
            id: 2,
            variant: Neuron(0.0, 0.0, 4)
        },
        Gene {
            weight: 0.3,
            id: 3,
            variant: Forward
        },
        Gene {
            weight: 0.7,
            id: 0,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.8,
            id: 1,
            variant: Input(0.0)
        },
        Gene {
            weight: 0.2,
            id: 0,
            variant: Recurrent
        },
    ];

    #[test]
    fn test_genome_is_correct() {
        if let Err(_) = pretty_env_logger::try_init() {
            // ignore error due to it being already initialized
        };

        let mut net = Network{
            size: TEST_GENOME.len() - 1,
            genome: TEST_GENOME.to_vec(),
            function: Activation::Linear
        };
        let output = net.evaluate(&vec![1.0, 1.0]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.654);
    }

    #[test]
    fn jump_recurrent_is_correct() {
        if let Err(_) = pretty_env_logger::try_init() {
            // ignore error due to it being already initialized
        };

        let mut net = Network{
            size: TEST_GENOME_2.len() - 1,
            genome: TEST_GENOME_2.to_vec(),
            function: Activation::Linear,
        };
        let output = net.evaluate(&vec![1.0, 1.0]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.49488);
    }

    #[test]
    fn test_recurrent_previous_value() {
        let genome = vec![
            Gene::neuron(1.0, 0, 2),
            Gene::recurrent(3.0, 1),
            Gene::neuron(1.0, 1, 1),
            Gene::bias(1.0),
        ];

        let mut net = Network{
            size: genome.len() - 1,
            genome,
            function: Activation::Linear,
        };
        // The recurrent jumper reads a previous value of zero despite the neuron already being
        // evaluated by the time the jumper is reached
        let output = net.evaluate(&[]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1.0);

        // The recurrent jumper now reads a non-zero previous value from the first evaluation
        let output2 = net.evaluate(&[]);
        assert_eq!(output2[0], 4.0);
    }
}
