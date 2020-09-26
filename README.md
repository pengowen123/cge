# cge

An implementation of the Common Genetic Encoding (direct encoding only). This library provides functionality for reading and writing neural networks to files and strings, and evaluating them. No method of training is provided, instead use [this](https://github.com/pengowen123/eant2). Train a neural network, save it to a file in the correct format, then load it using this library. The network can now be evaluated using the evaluate method.

# Usage

Add this to your Cargo.toml:

```
[dependencies]
cge = { git = "https://github.com/MathisWellmann/cge" }
```

And this to your crate root:

```rust
extern crate cge;
```

## TODOs:
- include activation function in gene rather than network wide
- add more activation functions
- add Activation::get_func(&self) -> fn(f64) -> f64
- resolve TODOs
- update README.md