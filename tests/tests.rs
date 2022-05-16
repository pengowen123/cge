extern crate cge;

use cge::Network;

const TEST_GENOME: &'static str = "0: n 1.0 0 3,n 1.0 1 2,n 1.0 3 2,i 1.0 0,i 1.0 1,i 1.0 1,n 1.0 2 4,f 1.0 3,i 1.0 0,i 1.0 1,r 1.0 0,b 1.0";

#[test]
fn test_network_eval() {
    let mut network = Network::from_str(TEST_GENOME).unwrap();

    let inputs = [1.0, 1.0];
    let result = network.evaluate(&inputs);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 8.0);
}
