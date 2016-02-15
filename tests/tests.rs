// File tests will fail unless the text files are in the right place
// Copy them to the crate root and the tests should pass

extern crate cge;

use cge::Network;
use cge::gene::Gene;
use cge::gene::GeneExtras::*;

const TEST_GENOME: [Gene; 11] = [
    Gene {
        weight: 1.0,
        id: 0,
        variant: Neuron(0.0, 2)
    },
    Gene {
        weight: 1.0,
        id: 1,
        variant: Neuron(0.0, 2)
    },
    Gene {
        weight: 1.0,
        id: 3,
        variant: Neuron(0.0, 2)
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
        variant: Neuron(0.0, 4)
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
    let network = Network::load_from_file("foo.txt").unwrap();
    let test_genome = TEST_GENOME.to_vec();
    
    assert_eq!(network.size, 10);
    assert!(equal(test_genome, network.genome));
}

#[test]
fn test_write_file() {
    let network = Network::load_from_file("foo.txt").unwrap();
    let test_genome = TEST_GENOME.to_vec();

    network.save_to_file("bar.txt").unwrap();

    let network = Network::load_from_file("bar.txt").unwrap();

    assert_eq!(network.size, 10);
    assert!(equal(test_genome, network.genome));
}

#[test]
fn test_network_eval() {
    let mut network = Network {
        size: 10,
        genome: TEST_GENOME.to_vec()
    };

    let inputs = vec![1.0, 1.0];
    let result = network.evaluate(&inputs);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 7.0);
}
