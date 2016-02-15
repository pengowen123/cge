/// An enum for storing additional information for different types of genes
<<<<<<< HEAD
#[derive(Clone, Debug, PartialEq)]
=======
#[derive(Clone, Debug)]
>>>>>>> 9f469de66585ebab2be137a99e708cbbbeb27db3
pub enum GeneExtras {
    /// Input contains a current value
    Input(f64),
    /// Neuron contains a current value and an input count
    Neuron(f64, usize),
    Forward,
    Recurrent,
    Bias
}

<<<<<<< HEAD
#[derive(Clone, Debug, PartialEq)]
=======
#[derive(Clone, Debug)]
>>>>>>> 9f469de66585ebab2be137a99e708cbbbeb27db3
pub struct Gene {
    pub weight: f64,
    pub id: usize,
    pub variant: GeneExtras
}

impl Gene {
    pub fn forward(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Forward
        }
    }

    pub fn recurrent(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Recurrent
        }
    }

    pub fn input(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Input(0.0)
        }
    }

    pub fn bias(weight: f64) -> Gene {
        Gene {
            weight: weight,
            id: 0,
            variant: GeneExtras::Bias
        }
    }

    pub fn neuron(weight: f64, id: usize, inputs: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Neuron(0.0, inputs)
        }
    }

    pub fn ref_input(&self) -> Option<(f64, usize, f64)> {
        if let GeneExtras::Input(ref weight) = self.variant {
            Some((self.weight, self.id, *weight))
        } else {
            None
        }
    }

    pub fn ref_neuron(&self) -> Option<(f64, usize, f64, usize)> {
        if let GeneExtras::Neuron(ref value, ref inputs) = self.variant {
            Some((self.weight, self.id, *value, *inputs))
        } else {
            None
        }
    }

    pub fn ref_mut_neuron<'a>(&'a mut self) -> Option<(&'a mut f64, &'a mut usize, &'a mut f64, &'a mut usize)> {
        if let GeneExtras::Neuron(ref mut value, ref mut inputs) = self.variant {
            Some((&mut self.weight, &mut self.id, value, inputs))
        } else {
            None
        }
    }
}
