extern crate cge;

use cge::Network;
use cge::gene::Gene;
use cge::gene::GeneExtras::*;
use cge::Activation;

const TEST_GENOME: [Gene; 12] = [
    Gene {
        weight: 1.0,
        id: 0,
        variant: Neuron(0.0, 0.0, 3)
    },
    Gene {
        weight: 1.0,
        id: 1,
        variant: Neuron(0.0, 0.0, 2)
    },
    Gene {
        weight: 1.0,
        id: 3,
        variant: Neuron(0.0, 0.0, 2)
    },
    Gene {
        weight: 1.0,
        id: 0,
        variant: Input(0.0)
    },
    Gene {
        weight: 1.0,
        id: 1,
        variant: Input(0.0)
    },
    Gene {
        weight: 1.0,
        id: 1,
        variant: Input(0.0)
    },
    Gene {
        weight: 1.0,
        id: 2,
        variant: Neuron(0.0, 0.0, 4)
    },
    Gene {
        weight: 1.0,
        id: 3,
        variant: Forward
    },
    Gene {
        weight: 1.0,
        id: 0,
        variant: Input(0.0)
    },
    Gene {
        weight: 1.0,
        id: 1,
        variant: Input(0.0)
    },
    Gene {
        weight: 1.0,
        id: 0,
        variant: Recurrent
    },
    Gene {
        weight: 1.0,
        id: 0,
        variant: Bias
    }
];

fn equal(a: Vec<Gene>, b: Vec<Gene>) -> bool {
    for i in 0..a.len() {
        let ga = &a[i];
        let gb = &b[i];

        if ga.weight != gb.weight ||
           ga.id != gb.id ||
           ga.variant != gb.variant {
               false;
        }
    }
    
    true
}
 
#[test]
fn test_read_file() {
    let network = Network::load_from_file("tests/foo.txt").unwrap();
    let test_genome = TEST_GENOME.to_vec();
    
    assert_eq!(network.size, 11);
    assert!(equal(test_genome, network.genome));
}

#[test]
fn test_write_file() {
    let network = Network::load_from_file("tests/foo.txt").unwrap();
    let test_genome = TEST_GENOME.to_vec();

    network.save_to_file("tests/bar.txt").unwrap();

    let network = Network::load_from_file("tests/bar.txt").unwrap();

    assert_eq!(network.size, 11);
    assert!(equal(test_genome, network.genome));
}

#[test]
fn test_network_eval() {
    let mut network = Network {
        size: 11,
        genome: TEST_GENOME.to_vec(),
        function: Activation::Linear
    };

    let inputs = [1.0, 1.0];
    let result = network.evaluate(&inputs);
    let result2 = network.evaluate(&[]);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 8.0);

    assert_eq!(result2.len(), 1);
    assert_eq!(result2[0], 9.0);
}
