use std::path::Path;
use std::fs::File;
use std::io::{Read, Write, Error, ErrorKind};
use std::io;

use crate::Network;
use crate::gene::*;
use crate::activation::*;

pub fn from_str(string: &str) -> Option<Network> {
    let parts = string.split(":").collect::<Vec<_>>();

    if parts.len() != 2 {
        return None;
    }

    let function = if let Ok(v) = parts[0].parse() {
        match v {
            0 => Activation::Linear,
            1 => Activation::Threshold,
            2 => Activation::Sign,
            3 => Activation::Sigmoid,
            _ => return None
        }
    } else {
        return None;
    };

    let string = parts[1];

    let genes = string.split(",");
    let mut genome = Vec::new();

    for gene in genes {
        let gene = gene.split_whitespace().collect::<Vec<&str>>();

        if gene.is_empty() {
            return None;
        }

        match gene[0] { 
            "n" => {
                if gene.len() != 4 {
                    return None;
                }

                genome.push(Gene::neuron(gene[1].parse::<f64>().unwrap(),
                                         gene[2].parse::<usize>().unwrap(),
                                         gene[3].parse::<usize>().unwrap()));
            },
            "i" => {
                if gene.len() != 3 {
                    return None;
                }

                genome.push(Gene::input(gene[1].parse::<f64>().unwrap(),
                                        gene[2].parse::<usize>().unwrap()));
            },
            "f" => {
                if gene.len() != 3 {
                    return None;
                }

                genome.push(Gene::forward(gene[1].parse::<f64>().unwrap(),
                                          gene[2].parse::<usize>().unwrap()));
            },
            "r" => {
                if gene.len() != 3 {
                    return None;
                }

                genome.push(Gene::recurrent(gene[1].parse::<f64>().unwrap(),
                                            gene[2].parse::<usize>().unwrap()));
            },
            "b" => {
                if gene.len() != 2 {
                    return None;
                }   
                
                genome.push(Gene::bias(gene[1].parse::<f64>().unwrap()));
            },
            _ => {
                return None;
            }
        }
    }

    if genome.is_empty() {
        return None;
    }

    Some(Network {
        size: genome.len() - 1,
        genome,
        function
    })
}

pub fn to_str(network: &Network) -> String {
    let mut data = format!("{}: ", network.function as i32);

    for gene in &network.genome {
        match gene.variant {
            GeneExtras::Input(_) => {
                data.push_str(&format!("i {} {},", gene.weight, gene.id));
            },
            GeneExtras::Neuron(_, ref inputs) => {
                data.push_str(&format!("n {} {} {},", gene.weight, gene.id, *inputs));
            },
            GeneExtras::Forward => {
                data.push_str(&format!("f {} {},", gene.weight, gene.id));
            },
            GeneExtras::Recurrent => {
                data.push_str(&format!("r {} {},", gene.weight, gene.id));
            },
            GeneExtras::Bias => {
                data.push_str(&format!("b {},", gene.weight));
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
        None => Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"))
    }
}

pub fn write_network(network: &Network, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = File::create(path)?;
    let data = to_str(network);
    
    file.write_all(data.as_bytes())
}
