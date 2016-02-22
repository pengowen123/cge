# cge

[![Build Status](https://travis-ci.org/pengowen123/cge.svg?branch=master)](https://travis-ci.org/pengowen123/cge)

An implementation of the Common Genetic Encoding (direct encoding only). This library provides functionality for reading and writing neural networks to files and strings, and evaluating them. No method of training is provided, instead use [this](https://github.com/pengowen123/eant2). Train a neural network, save it to a file in the correct format, then load it using this library. The network can now be evaluated using the evaluate method.

# Usage

Add this to your Cargo.toml:

```
[dependencies]
cge = { git = "https://github.com/pengowen123/cge" }
```

And this to your crate root:

```rust
extern crate cge;
```

See the [documentation](http://pengowen123.github.io/cge/cge/index.html) for complete instructions.
