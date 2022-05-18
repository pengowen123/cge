use std::fs::File;
use std::io;
use std::io::{Error, ErrorKind, Read, Write};
use std::path::Path;

use crate::activation::*;
use crate::gene::*;
use crate::Network;

pub fn from_str(string: &str) -> Option<Network> {
    let parts = string.split(":").collect::<Vec<_>>();

    if parts.len() != 2 {
        return None;
    }

    let activation = if let Ok(v) = parts[0].parse() {
        match v {
            0 => Activation::Linear,
            1 => Activation::UnitStep,
            2 => Activation::Sign,
            3 => Activation::Sigmoid,
            4 => Activation::Tanh,
            5 => Activation::SoftSign,
            6 => Activation::BentIdentity,
            7 => Activation::Relu,
            _ => return None,
        }
    } else {
        return None;
    };

    let string = parts[1];

    let genes = string.split(",");
    let mut genome: Vec<Gene> = Vec::new();

    for gene in genes {
        let gene = gene.split_whitespace().collect::<Vec<&str>>();

        if gene.is_empty() {
            return None;
        }

        let new_gene;

        match gene[0] {
            "n" => {
                if gene.len() != 4 {
                    return None;
                }

                let weight = gene[1].parse::<f64>().unwrap();
                let id = gene[2].parse::<usize>().unwrap();
                let inputs = gene[3].parse::<usize>().unwrap();
                let neuron = Neuron::new(NeuronId::new(id), inputs, weight);
                new_gene = neuron.into();
            }
            "i" => {
                if gene.len() != 3 {
                    return None;
                }

                let weight = gene[1].parse::<f64>().unwrap();
                let input_id = gene[2].parse::<usize>().unwrap();
                let input = Input::new(InputId::new(input_id), weight);
                new_gene = input.into();
            }
            "f" => {
                if gene.len() != 3 {
                    return None;
                }

                let weight = gene[1].parse::<f64>().unwrap();
                let source_id = gene[2].parse::<usize>().unwrap();
                let forward = ForwardJumper::new(NeuronId::new(source_id), weight);
                new_gene = forward.into();
            }
            "r" => {
                if gene.len() != 3 {
                    return None;
                }

                let weight = gene[1].parse::<f64>().unwrap();
                let source_id = gene[2].parse::<usize>().unwrap();
                let recurrent = RecurrentJumper::new(NeuronId::new(source_id), weight);
                new_gene = recurrent.into();
            }
            "b" => {
                if gene.len() != 2 {
                    return None;
                }

                let value = gene[1].parse::<f64>().unwrap();
                let bias = Bias::new(value);
                new_gene = bias.into();
            }
            _ => {
                return None;
            }
        }

        genome.push(new_gene);
    }

    if genome.is_empty() {
        return None;
    }

    Some(Network::new(genome, activation).unwrap())
}

pub fn to_str(network: &Network) -> String {
    let mut data = format!("{}: ", network.activation() as i32);

    for gene in network.genome() {
        match gene {
            Gene::Input(input) => {
                data.push_str(&format!("i {} {},", input.weight(), input.id().as_usize()));
            }
            Gene::Neuron(neuron) => {
                data.push_str(&format!(
                    "n {} {} {},",
                    neuron.weight(),
                    neuron.id().as_usize(),
                    neuron.num_inputs()
                ));
            }
            Gene::ForwardJumper(forward) => {
                data.push_str(&format!(
                    "f {} {},",
                    forward.weight(),
                    forward.source_id().as_usize()
                ));
            }
            Gene::RecurrentJumper(recurrent) => {
                data.push_str(&format!(
                    "r {} {},",
                    recurrent.weight(),
                    recurrent.source_id().as_usize()
                ));
            }
            Gene::Bias(bias) => {
                data.push_str(&format!("b {},", bias.value()));
            }
        }
    }

    data.pop();
    data
}

pub fn read_network(path: &str) -> io::Result<Network> {
    let path = Path::new(path);
    let mut file = File::open(path)?;
    let mut data = String::new();

    file.read_to_string(&mut data)?;

    let network = from_str(&data);

    match network {
        Some(n) => Ok(n),
        None => Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid neural network file format",
        )),
    }
}

pub fn write_network(network: &Network, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = File::create(path)?;
    let data = to_str(network);

    file.write_all(data.as_bytes())
}
