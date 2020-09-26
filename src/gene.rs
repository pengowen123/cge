/// An enum for storing additional information for different types of genes
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeneExtras {
    /// Input contains a current value
    Input(f64),
    /// Neuron contains a current value and an input count
    Neuron(f64, usize),
    Forward,
    Recurrent,
    Bias
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gene {
    pub weight: f64,
    pub id: usize,
    pub variant: GeneExtras
}

impl Gene {
    pub fn forward(weight: f64, id: usize) -> Gene {
        Gene {
            weight,
            id,
            variant: GeneExtras::Forward
        }
    }

    pub fn recurrent(weight: f64, id: usize) -> Gene {
        Gene {
            weight,
            id,
            variant: GeneExtras::Recurrent
        }
    }

    pub fn input(weight: f64, id: usize) -> Gene {
        Gene {
            weight,
            id,
            variant: GeneExtras::Input(0.0)
        }
    }

    pub fn bias(weight: f64) -> Gene {
        Gene {
            weight,
            id: 0,
            variant: GeneExtras::Bias
        }
    }

    pub fn neuron(weight: f64, id: usize, inputs: usize) -> Gene {
        Gene {
            weight,
            id,
            variant: GeneExtras::Neuron(0.0, inputs)
        }
    }

    #[doc(hidden)]
    pub fn ref_input(&self) -> Option<(f64, usize, f64)> {
        if let GeneExtras::Input(ref weight) = self.variant {
            Some((self.weight, self.id, *weight))
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn ref_neuron(&self) -> Option<(f64, usize, f64, usize)> {
        if let GeneExtras::Neuron(ref value, ref inputs) = self.variant {
            Some((self.weight, self.id, *value, *inputs))
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn ref_mut_neuron(&mut self) -> Option<(&mut f64, &mut usize, &mut f64, &mut usize)> {
        if let GeneExtras::Neuron(ref mut value, ref mut inputs) = self.variant {
            Some((&mut self.weight, &mut self.id, value, inputs))
        } else {
            None
        }
    }
}
