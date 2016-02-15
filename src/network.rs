//! The neural network struct. See the module level documentation for usage.

use std::ops::Range;
use std::io;

use utils::Stack;
use file;
use gene::*;
use gene::GeneExtras::*;

const BIAS_GENE_VALUE: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct Network {
    // size should be the length of the genome minus one, don't forget
    pub size: usize,
    pub genome: Vec<Gene>
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
    /// ```
    /// use cge::Network;
    ///
    /// // Get the output of the neural network the the specified inputs
    /// let result = network.evaluate(vec![1.0, 1.0]);
    ///
    /// // Get the output of the neural network with no inputs
    /// let result = network.evaluate(Vec::new());
    ///
    /// // Get the output of the neural network with too many inputs (extras aren't used)
    /// let result = network.evaluate(vec![1.0, 1.0, 1.0]);
    ///
    /// 
    /// // Let's say adder.ann is a file with a neural network with recurrent connections, used for
    /// // adding numbers together.
    /// let mut adder = Network::load_from_file("adder.ann");
    ///
    /// // result_one will be 1.0
    /// let result_one = adder.evaluate(vec![1.0]);
    ///
    /// // result_two will be 3.0
    /// let result_two = adder.evaluate(vec![2.0]);
    ///
    /// // result_three will be 5.0
    /// let result_three = adder.evaluate(vec![2.0]);
    ///
    /// // If this behavior is not desired, call the clear_state method between evaluations:
    /// let result_one = adder.evaluate(vec![1.0]);
    /// 
    /// adder.clear_state();
    ///
    /// // The 1.0 from the previous call is gone, so result_two will be 2.0
    /// let result_two = adder.evaluate(vec![2.0]);
    /// ```
    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // Set inputs
        for gene in &mut self.genome {
            if let Input(ref mut current_value) = gene.variant {
                *current_value = match inputs.get(gene.id) {
                    Some(v) => *v,
                    None => 0.0
                }
            }
        }

        let size = self.size;
        self.evaluate_slice(0..size, true)
    }

    /// Clears the internal state of the neural network.
    pub fn clear_state(&mut self) {
        for gene in &mut self.genome {
            match gene.variant {
                Input(ref mut current_value) => {
                    *current_value = 0.0;
                },
                Neuron(ref mut current_value, _) => {
                    *current_value = 0.0;
                }
                _ => {}
            }
        }
    }

    /// Saves the neural network to a file. Returns an empty tuple on success, or an io error.
    ///
    /// # Examples
    ///
    /// ```
    /// // Save network to neural_network.ann
    /// network.save_to_file("neural_network.ann");
    /// ```
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        file::write_network(self, path)
    }

    /// Loads a neural network from a file. No guarantees are made about the validity of the
    /// genome. Returns the network, or an io error. If the file is in a bad format,
    /// io::ErrorKind::InvalidData is returned.
    /// 
    /// # Examples
    ///
    /// ```
    /// use cge::Network;
    ///
    /// // Loads a network from the file neural_network.ann
    /// let mut network = Network::load_from_file("neural_network.ann").unwrap();
    /// ```
    pub fn load_from_file(path: &str) -> io::Result<Network> {
        file::read_network(path)
    }

    // Returns the output of sub-linear genome in the given range
    fn evaluate_slice(&mut self, range: Range<usize>, neuron_update: bool) -> Vec<f64> {
        let mut gene_index = range.end;
        // Initialize a stack for evaluating the neural network
        let mut stack = Stack::new();
        
        // Iterate backwards over the specified slice
        while gene_index >= range.start {
            let variant = self.genome[gene_index].variant.clone();

            // For debugging: uncomment these print statements
            // Perhaps add it as an option to help users
            // println!("{:?}", variant);

            match variant {
                Input(_) => {
                    // If the gene is an input, push its value multiplied by the inputs weight onto
                    // the stack
                    let (weight, _, value) = self.genome[gene_index].ref_input().unwrap();
                    stack.push(weight * value);
                },
                Neuron(_, _) => {
                    // If the gene is a neuron, pop a number (the neurons input count) of inputs
                    // off the stack, and push their sum multiplied by the neurons weight onto the
                    // stack
                    let (weight, _, value, inputs) = self.genome[gene_index].ref_mut_neuron().unwrap();
                    let new_value = stack.pop(*inputs).unwrap().iter().fold(0.0, |acc, i| acc + i);

                    if neuron_update {
                        *value = new_value;
                    }

                    stack.push(*weight * new_value);
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
                    let subnetwork_range = self.get_subnetwork_index(id).unwrap();

                    let result = self.evaluate_slice(subnetwork_range, false);
                    // println!("{:?}", result);
                    stack.push(weight * result[0]);
                },
                Recurrent => {
                    // If the gene is a recurrent jumper, push the previous value of the neuron
                    // with the id of the jumper multiplied by the jumpers weight onto the stack
                    let gene = &self.genome[gene_index];
                    let neuron = &self.genome[self.get_neuron_index(gene.id).unwrap()];
                    
                    if let Neuron(ref current_value, _) = neuron.variant {
                        stack.push(gene.weight * *current_value);
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
            // println!("{:?}", stack.data);
        }

        stack.data
    }

    // Returns the start and end index of the subnetwork starting at the neuron with the given id,
    // or None if it does not exist
    fn get_subnetwork_index(&self, id: usize) -> Option<Range<usize>> {
        let start = match self.get_neuron_index(id) {
            Some(i) => i,
            None => return None
        };

        let mut end = start;
        let mut sum = 0;

        // Iterate through genes after the start index, modifying the sum each step 
        // I could use an iterator here, but it would be messy
        for gene in &self.genome[start..self.size] {
            match gene.variant {
                Neuron(_, ref inputs) => {
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
                start: start,
                end: end
            })
        }
    }

    // Returns the index of the neuron with the given id, or None if it does not exist
    fn get_neuron_index(&self, id: usize) -> Option<usize> {
        let mut result = None;

        for (i, gene) in self.genome.iter().enumerate() {
            if let Neuron(_, _) = gene.variant {
                if gene.id == id {
                    result = Some(i);
                }
            }
        }

        result
    }
}
