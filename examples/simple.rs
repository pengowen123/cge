//! An example of loading, using, and saving a network. Requires the `serde_json` feature.

use cge::gene::{Bias, NeuronId};
use cge::{Network, WithRecurrentState};

use serde::{Deserialize, Serialize};

// Custom data to be stored alongside the saved network
#[derive(Debug, Serialize, Deserialize)]
struct Foo {
    x: i32,
    y: [f64; 2],
}

fn main() {
    // Load the network from a file of any version
    let (mut network, mut metadata, extra) = Network::<f64>::load_file::<Foo, _>(
        "test_data/with_extra_data_v1.cge",
        WithRecurrentState(true),
    )
    .unwrap();

    println!("metadata: {:?}", metadata);
    println!("extra: {:?}", extra);

    // Use the network
    let output = network.evaluate(&[1.0, 2.0]).unwrap();
    println!("output: {:?}", output);

    network.clear_state();

    let output_2 = network.evaluate(&[1.0, 2.0]).unwrap();
    println!("output 2: {:?}", output_2);

    // Modify the network
    let new_gene = Bias::new(1.5);
    network.add_non_neuron(NeuronId::new(2), new_gene).unwrap();

    // Save the network to a file of the latest version
    // A different version may be specified by using a more specific metadata type here (e.g.,
    // encoding::v1::Metadata or just `metadata.into_v1()`)
    metadata.description = Some("a new description".into());
    let metadata = metadata.into_latest_version().unwrap();
    network
        .to_file(
            metadata,
            extra,
            WithRecurrentState(true),
            "test_output/with_extra_data.cge",
            true,
        )
        .unwrap();
}
