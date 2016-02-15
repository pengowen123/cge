use std::path::Path;
use std::fs::File;
use std::io::{Read, Write, Error, ErrorKind};
use std::io;

use Network;
use gene::*;

pub fn read_network(path: &str) -> io::Result<Network> {
    let path = Path::new(path);
    let mut file = try!(File::open(path));
    let mut data = String::new();

    try!(file.read_to_string(&mut data));

    let genes = data.split(",");
    let mut genome = Vec::new();

    for gene in genes {
        let gene = gene.split_whitespace().collect::<Vec<&str>>();

        if gene.is_empty() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
        }

        match gene[0] { 
            "n" => {
                if gene.len() != 4 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
                }

                genome.push(Gene::neuron(gene[1].parse::<f64>().unwrap(),
                                         gene[2].parse::<usize>().unwrap(),
                                         gene[3].parse::<usize>().unwrap()));
            },
            "i" => {
                if gene.len() != 3 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
                }

                genome.push(Gene::input(gene[1].parse::<f64>().unwrap(),
                                        gene[2].parse::<usize>().unwrap()));
            },
            "f" => {
                if gene.len() != 3 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
                }

                genome.push(Gene::forward(gene[1].parse::<f64>().unwrap(),
                                          gene[2].parse::<usize>().unwrap()));
            },
            "r" => {
                if gene.len() != 3 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
                }

                genome.push(Gene::recurrent(gene[1].parse::<f64>().unwrap(),
                                            gene[2].parse::<usize>().unwrap()));
            },
            "b" => {
                if gene.len() != 2 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
                }

                genome.push(Gene::bias(gene[1].parse::<f64>().unwrap()));
            }
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
            }
        }
    }

    if genome.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid neural network file format"));
    }

    Ok(Network {
        size: genome.len() - 1,
        genome: genome
    })
}

pub fn write_network(network: &Network, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = try!(File::create(path));
    let mut data = String::new();

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
                data.push_str(&format!("b {}", gene.weight));
            }
        }
    }

    data.pop();
    file.write_all(data.as_bytes())
}
