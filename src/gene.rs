//! Different types of genes that can be used in a network genome.

use serde::{Deserialize, Serialize};

/// A bias gene.
///
/// Adds a constant value to the network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bias {
    value: f64,
}

impl Bias {
    /// Returns a new `Bias` that adds a constant `value` to the network.
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    /// Returns the value of the `Bias`.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// The ID of a network input.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputId(usize);

impl InputId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// An input gene.
///
/// Adds a connection to one of the network inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    // The ID of the network input referred to
    id: InputId,
    weight: f64,
}

impl Input {
    /// Returns a new `Input` that connects to the network input with the id and weights it by
    /// `weight`.
    pub fn new(id: InputId, weight: f64) -> Self {
        Self { id, weight }
    }

    /// Returns the id of the network input this `Input` refers to.
    pub fn id(&self) -> InputId {
        self.id
    }

    /// Returns the weight of this `Input`.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// The ID of a neuron in a network.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NeuronId(usize);

impl NeuronId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// A neuron gene.
///
/// Takes some number of incoming connections and applies the activation function to their sum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Neuron {
    // The ID of this neuron
    id: NeuronId,
    // The number of incoming connections to this neuron
    num_inputs: usize,
    // The weight to apply to the result of the activation function
    // Note that this weight is not used when the neuron is referred to by a jumper connection; the
    // jumper's weight is used instead
    weight: f64,
    // The unweighted value outputted by this neuron during the current network evaluation if it has
    // been calculated already
    #[serde(skip)]
    current_value: Option<f64>,
    // The unweighted value outputted by this neuron during the previous network evaluation
    #[serde(skip)]
    previous_value: f64,
}

impl Neuron {
    /// Returns a new `Neuron` that takes `num_inputs` inputs and weights its output by `weight`.
    pub fn new(id: NeuronId, num_inputs: usize, weight: f64) -> Self {
        Self {
            id,
            num_inputs,
            weight,
            current_value: None,
            previous_value: 0.0,
        }
    }

    /// Returns the id of this `Neuron`.
    pub fn id(&self) -> NeuronId {
        self.id
    }

    /// Returns the number of inputs required by this `Neuron`.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Sets the number of inputs required by this `Neuron`.
    pub(crate) fn set_num_inputs(&mut self, num_inputs: usize) {
        self.num_inputs = num_inputs;
    }

    /// Returns the weight of this `Neuron`.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub(crate) fn current_value(&self) -> Option<f64> {
        self.current_value
    }

    pub(crate) fn set_current_value(&mut self, value: Option<f64>) {
        self.current_value = value;
    }

    pub(crate) fn previous_value(&self) -> f64 {
        self.previous_value
    }

    pub(crate) fn set_previous_value(&mut self, value: f64) {
        self.previous_value = value;
    }
}

/// A forward jumper gene.
///
/// Adds a connection to the output of a source neuron with a higher depth than the parent neuron
/// of the jumper.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForwardJumper {
    // The ID of the source neuron
    source_id: NeuronId,
    // The weight of the forward jumper connection
    // This replaces the weight of the source neuron
    weight: f64,
}

impl ForwardJumper {
    /// Returns a new `ForwardJumper` that connects to the output of the neuron with the id and
    /// weights it by `weight`.
    pub fn new(source_id: NeuronId, weight: f64) -> Self {
        Self { source_id, weight }
    }

    /// Returns the id of the source neuron of this `ForwardJumper`.
    pub fn source_id(&self) -> NeuronId {
        self.source_id
    }

    /// Returns the weight of this `ForwardJumper`.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// A recurrent jumper gene.
///
/// Adds a connection to the output from the previous network evaluation of a source neuron with
/// any depth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecurrentJumper {
    // The ID of the source neuron
    source_id: NeuronId,
    // The weight of the forward jumper connection
    // This replaces the weight of the source neuron
    weight: f64,
}

impl RecurrentJumper {
    /// Returns a new `RecurrentJumper` that connects to the output of the neuron with the id and
    /// weights it by `weight`.
    pub fn new(source_id: NeuronId, weight: f64) -> Self {
        Self { source_id, weight }
    }

    /// Returns the id of the source neuron of this `ForwardJumper`.
    pub fn source_id(&self) -> NeuronId {
        self.source_id
    }

    /// Returns the weight of this `RecurrentJumper`.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// A single gene in a genome, which can be either a [`Bias`], [`Input`], [`Neuron`],
/// [`ForwardJumper`], or [`RecurrentJumper`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[serde(rename_all = "lowercase")]
pub enum Gene {
    /// See [`Bias`].
    Bias(Bias),
    /// See [`Input`].
    Input(Input),
    /// See [`Neuron`].
    Neuron(Neuron),
    /// See [`ForwardJumper`].
    ForwardJumper(ForwardJumper),
    /// See [`RecurrentJumper`].
    RecurrentJumper(RecurrentJumper),
}

impl Gene {
    /// Returns whether this is a [`Bias`] gene.
    pub fn is_bias(&self) -> bool {
        matches!(self, Self::Bias(_))
    }

    /// Returns whether this is a [`Input`] gene.
    pub fn is_input(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    /// Returns whether this is a [`Neuron`] gene.
    pub fn is_neuron(&self) -> bool {
        matches!(self, Self::Neuron(_))
    }

    /// Returns whether this is a [`ForwardJumper`] gene.
    pub fn is_forward_jumper(&self) -> bool {
        matches!(self, Self::ForwardJumper(_))
    }

    /// Returns whether this is a [`RecurrentJumper`] gene.
    pub fn is_recurrent_jumper(&self) -> bool {
        matches!(self, Self::RecurrentJumper(_))
    }

    /// Returns a reference to the contained [`Bias`] if this is a bias gene.
    pub fn as_bias(&self) -> Option<&Bias> {
        if let Self::Bias(bias) = self {
            Some(bias)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`Input`] if this is an input gene.
    pub fn as_input(&self) -> Option<&Input> {
        if let Self::Input(input) = self {
            Some(input)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`Neuron`] if this is a neuron gene.
    pub fn as_neuron(&self) -> Option<&Neuron> {
        if let Self::Neuron(neuron) = self {
            Some(neuron)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`ForwardJumper`] if this is a forward jumper gene.
    pub fn as_forward_jumper(&self) -> Option<&ForwardJumper> {
        if let Self::ForwardJumper(forward) = self {
            Some(forward)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`RecurrentJumper`] if this is a recurrent jumper gene.
    pub fn as_recurrent_jumper(&self) -> Option<&RecurrentJumper> {
        if let Self::RecurrentJumper(recurrent) = self {
            Some(recurrent)
        } else {
            None
        }
    }

    /// Returns a mutable reference to the contained [`Neuron`] if this is a neuron gene.
    pub(crate) fn as_mut_neuron(&mut self) -> Option<&mut Neuron> {
        if let Self::Neuron(neuron) = self {
            Some(neuron)
        } else {
            None
        }
    }
}

impl From<Bias> for Gene {
    fn from(bias: Bias) -> Self {
        Self::Bias(bias)
    }
}

impl From<Input> for Gene {
    fn from(input: Input) -> Self {
        Self::Input(input)
    }
}

impl From<Neuron> for Gene {
    fn from(neuron: Neuron) -> Self {
        Self::Neuron(neuron)
    }
}

impl From<ForwardJumper> for Gene {
    fn from(forward: ForwardJumper) -> Self {
        Self::ForwardJumper(forward)
    }
}

impl From<RecurrentJumper> for Gene {
    fn from(recurrent: RecurrentJumper) -> Self {
        Self::RecurrentJumper(recurrent)
    }
}

/// Like [`Gene`], but cannot be a [`Neuron`] gene.
#[derive(Clone, Debug, PartialEq)]
pub enum NonNeuronGene {
    /// See [`Bias`].
    Bias(Bias),
    /// See [`Input`].
    Input(Input),
    /// See [`ForwardJumper`].
    ForwardJumper(ForwardJumper),
    /// See [`RecurrentJumper`].
    RecurrentJumper(RecurrentJumper),
}

impl From<Bias> for NonNeuronGene {
    fn from(bias: Bias) -> Self {
        Self::Bias(bias)
    }
}

impl From<Input> for NonNeuronGene {
    fn from(input: Input) -> Self {
        Self::Input(input)
    }
}

impl From<ForwardJumper> for NonNeuronGene {
    fn from(forward: ForwardJumper) -> Self {
        Self::ForwardJumper(forward)
    }
}

impl From<RecurrentJumper> for NonNeuronGene {
    fn from(recurrent: RecurrentJumper) -> Self {
        Self::RecurrentJumper(recurrent)
    }
}

impl From<NonNeuronGene> for Gene {
    fn from(gene: NonNeuronGene) -> Self {
        match gene {
            NonNeuronGene::Bias(bias) => Self::Bias(bias),
            NonNeuronGene::Input(input) => Self::Input(input),
            NonNeuronGene::ForwardJumper(forward) => Self::ForwardJumper(forward),
            NonNeuronGene::RecurrentJumper(recurrent) => Self::RecurrentJumper(recurrent),
        }
    }
}
