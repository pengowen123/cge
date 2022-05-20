//! An example of saving and loading a network to/from a file.

use cge::Network;

use serde::{Deserialize, Serialize};

// Custom data to be stored alongside the saved network
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Foo {
    x: i32,
    y: [f64; 2],
}

fn main() {
    // Load the network from a file of any version
    let (mut network, mut metadata, extra) =
        Network::load_file::<Foo, _>("test_data/with_extra_data_v1.cge").unwrap();

    println!("metadata: {:?}", metadata);
    println!("extra: {:?}", extra);

    // Use the network
    let output = network.evaluate(&[1.0, 2.0]).unwrap();
    println!("output: {:?}", output);

    // Save the network to a file of the latest version
    // A different version may be specified by using a more specific metadata type here (e.g.,
    // encoding::v1::Metadata or just `metadata.into_v1()`)
    metadata.description = "a new description".to_string().into();
    let metadata = metadata.into_latest_version().unwrap();
    network
        .to_file(metadata, extra, "test_output/with_extra_data.cge", true)
        .unwrap();
}
