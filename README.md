# cge

[![Crates.io](https://img.shields.io/crates/v/cge)](https://crates.io/crates/cge)
[![Rust](https://github.com/pengowen123/cge/actions/workflows/rust.yml/badge.svg?branch=master)](https://github.com/pengowen123/cge/actions/workflows/rust.yml)

A Rust library for creating, using, and modifying artificial neural networks using the [Common Genetic Encoding
(CGE)][0]. See [`const_cge`][1] for a similar library geared towards embedded environments and performance-critical use cases. For the creation of CGE-compatible neural networks, see the [`eant2`][2] library.

Requires Rust 1.43 or later.

## Features

`cge` is intended to be a complete solution for interacting with CGE neural networks in the direct encoding case only. It currently provides these features:

- Loading and saving neural networks in a backwards-compatible and extensible format from files and strings
- Evaluation of neural networks as well as saving, loading, and resetting their internal states
- Modification of neural networks through structural and weight mutations
- Genome and mutation validity checking

## Quick Start

Add this to your Cargo.toml:

```
[dependencies]
cge = "0.1"
```

Then, to load and use an existing neural network from a file:

```rust
use cge::{Network, WithRecurrentState};

// `extra` is any user-defined data stored alongside the network
let (mut network, metadata, extra) =
    Network::<f64>::load_file::<(), _>("network.cge", WithRecurrentState(true)).unwrap();

println!("description: {:?}", metadata.description);
println!("num inputs, outputs: {}, {}", network.num_inputs(), network.num_outputs());
println!("{:?}", network.evaluate(&[1.0, 2.0]).unwrap());

network.clear_state();

println!("{:?}", network.evaluate(&[2.0, 0.0]).unwrap());
```

For more information, see the [documentation][3] and [examples][4].

## Contributing

Contributions are welcome! You can contribute by reporting any bugs or issues you have with the library, adding documentation, fixing bugs, or adding features.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without any additional terms or conditions.

## License

Licensed under either of

    Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
    MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

[0]: https://dl.acm.org/doi/10.1145/1276958.1277162
[1]: https://github.com/wbrickner/const_cge
[2]: https://github.com/pengowen123/eant2
[3]: https://docs.rs/cge/latest/cge
[4]: https://github.com/pengowen123/cge/tree/master/examples
