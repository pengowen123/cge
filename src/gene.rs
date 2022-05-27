//! Different types of genes that can be used in a [`Network`][crate::Network] genome.

use num_traits::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A bias gene.
///
/// Adds a constant value to the [`Network`][crate::Network].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bias<T> {
    value: T,
}

impl<T: Float> Bias<T> {
    /// Returns a new `Bias` that adds a constant `value` to the [`Network`][crate::Network].
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Returns the value of the `Bias`.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns a mutable reference to the value of the `Bias`.
    pub fn mut_value(&mut self) -> &mut T {
        &mut self.value
    }
}

/// The ID of a [`Network`][crate::Network]'s [`Input`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InputId(usize);

impl InputId {
    /// Returns a new `InputId` with the given id.
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns this `InputId` as a `usize`.
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// An input gene.
///
/// Adds a connection to one of the [`Network`][crate::Network] inputs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Input<T> {
    // The ID of the network input referred to
    id: InputId,
    weight: T,
}

impl<T: Float> Input<T> {
    /// Returns a new `Input` that connects to the [`Network`][crate::Network] input with the id
    /// and weights it by `weight`.
    pub fn new(id: InputId, weight: T) -> Self {
        Self { id, weight }
    }

    /// Returns the id of the [`Network`][crate::Network] input this `Input` refers to.
    pub fn id(&self) -> InputId {
        self.id
    }

    /// Returns the weight of this `Input`.
    pub fn weight(&self) -> T {
        self.weight
    }

    /// Returns a mutable reference to the weight of this `Input`.
    pub fn mut_weight(&mut self) -> &mut T {
        &mut self.weight
    }
}

/// The ID of a [`Neuron`] in a [`Network`][crate::Network].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronId(usize);

impl NeuronId {
    /// Returns a new `NeuronId` with the given id.
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns this `NeuronId` as a `usize`.
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// A neuron gene.
///
/// Takes some number of incoming connections and applies the activation function to their sum.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neuron<T: Float> {
    // The ID of this neuron
    id: NeuronId,
    // The number of incoming connections to this neuron
    num_inputs: usize,
    // The weight to apply to the result of the activation function
    // Note that this weight is not used when the neuron is referred to by a jumper connection; the
    // jumper's weight is used instead
    weight: T,
    // The unweighted value outputted by this neuron during the current network evaluation if it has
    // been calculated already
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(feature = "serde", serde(default = "Default::default"))]
    current_value: Option<T>,
    // The unweighted value outputted by this neuron during the previous network evaluation
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(feature = "serde", serde(default = "T::zero"))]
    previous_value: T,
}

impl<T: Float> Neuron<T> {
    /// Returns a new `Neuron` that takes `num_inputs` inputs and weights its output by `weight`.
    pub fn new(id: NeuronId, num_inputs: usize, weight: T) -> Self {
        Self {
            id,
            num_inputs,
            weight,
            current_value: None,
            previous_value: T::zero(),
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
    pub fn weight(&self) -> T {
        self.weight
    }

    /// Returns a mutable reference to the weight of this `Neuron`.
    pub(crate) fn mut_weight(&mut self) -> &mut T {
        &mut self.weight
    }

    pub(crate) fn current_value(&self) -> Option<T> {
        self.current_value
    }

    pub(crate) fn set_current_value(&mut self, value: Option<T>) {
        self.current_value = value;
    }

    pub(crate) fn previous_value(&self) -> T {
        self.previous_value
    }

    pub(crate) fn mut_previous_value(&mut self) -> &mut T {
        &mut self.previous_value
    }
}

/// A forward jumper gene.
///
/// Adds a connection to the output of a source neuron with a higher depth than the parent neuron
/// of the jumper.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ForwardJumper<T> {
    // The ID of the source neuron
    source_id: NeuronId,
    // The weight of the forward jumper connection
    // This replaces the weight of the source neuron
    weight: T,
}

impl<T: Float> ForwardJumper<T> {
    /// Returns a new `ForwardJumper` that connects to the output of the neuron with the id and
    /// weights it by `weight`.
    pub fn new(source_id: NeuronId, weight: T) -> Self {
        Self { source_id, weight }
    }

    /// Returns the id of the source neuron of this `ForwardJumper`.
    pub fn source_id(&self) -> NeuronId {
        self.source_id
    }

    /// Returns the weight of this `ForwardJumper`.
    pub fn weight(&self) -> T {
        self.weight
    }

    /// Returns a mutable reference to the weight of this `ForwardJumper`.
    pub fn mut_weight(&mut self) -> &mut T {
        &mut self.weight
    }
}

/// A recurrent jumper gene.
///
/// Adds a connection to the output from the previous [`Network`][crate::Network] evaluation of a
/// source [`Neuron`] with any depth.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecurrentJumper<T> {
    // The ID of the source neuron
    source_id: NeuronId,
    // The weight of the forward jumper connection
    // This replaces the weight of the source neuron
    weight: T,
}

impl<T: Float> RecurrentJumper<T> {
    /// Returns a new `RecurrentJumper` that connects to the output of the neuron with the id and
    /// weights it by `weight`.
    pub fn new(source_id: NeuronId, weight: T) -> Self {
        Self { source_id, weight }
    }

    /// Returns the id of the source neuron of this `ForwardJumper`.
    pub fn source_id(&self) -> NeuronId {
        self.source_id
    }

    /// Returns the weight of this `RecurrentJumper`.
    pub fn weight(&self) -> T {
        self.weight
    }

    /// Returns a mutable reference to the weight of this `RecurrentJumper`.
    pub fn mut_weight(&mut self) -> &mut T {
        &mut self.weight
    }
}

/// A single gene in a genome, which can be either a [`Bias`], [`Input`], [`Neuron`],
/// [`ForwardJumper`], or [`RecurrentJumper`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind"))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Gene<T: Float> {
    /// See [`Bias`].
    Bias(Bias<T>),
    /// See [`Input`].
    Input(Input<T>),
    /// See [`Neuron`].
    Neuron(Neuron<T>),
    /// See [`ForwardJumper`].
    ForwardJumper(ForwardJumper<T>),
    /// See [`RecurrentJumper`].
    RecurrentJumper(RecurrentJumper<T>),
}

impl<T: Float> Gene<T> {
    /// Returns the weight of this `Gene` or its value if it is a [`Bias`].
    pub fn weight(&self) -> T {
        match self {
            Self::Bias(bias) => bias.value(),
            Self::Input(input) => input.weight(),
            Self::Neuron(neuron) => neuron.weight(),
            Self::ForwardJumper(forward) => forward.weight(),
            Self::RecurrentJumper(recurrent) => recurrent.weight(),
        }
    }

    /// Returns a mutable reference to the weight of this `Gene` or its value if it is a [`Bias`].
    pub(crate) fn mut_weight(&mut self) -> &mut T {
        match self {
            Self::Bias(bias) => bias.mut_value(),
            Self::Input(input) => input.mut_weight(),
            Self::Neuron(neuron) => neuron.mut_weight(),
            Self::ForwardJumper(forward) => forward.mut_weight(),
            Self::RecurrentJumper(recurrent) => recurrent.mut_weight(),
        }
    }

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
    pub fn as_bias(&self) -> Option<&Bias<T>> {
        if let Self::Bias(bias) = self {
            Some(bias)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`Input`] if this is an input gene.
    pub fn as_input(&self) -> Option<&Input<T>> {
        if let Self::Input(input) = self {
            Some(input)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`Neuron`] if this is a neuron gene.
    pub fn as_neuron(&self) -> Option<&Neuron<T>> {
        if let Self::Neuron(neuron) = self {
            Some(neuron)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`ForwardJumper`] if this is a forward jumper gene.
    pub fn as_forward_jumper(&self) -> Option<&ForwardJumper<T>> {
        if let Self::ForwardJumper(forward) = self {
            Some(forward)
        } else {
            None
        }
    }

    /// Returns a reference to the contained [`RecurrentJumper`] if this is a recurrent jumper gene.
    pub fn as_recurrent_jumper(&self) -> Option<&RecurrentJumper<T>> {
        if let Self::RecurrentJumper(recurrent) = self {
            Some(recurrent)
        } else {
            None
        }
    }

    /// Returns a mutable reference to the contained [`Neuron`] if this is a neuron gene.
    pub(crate) fn as_mut_neuron(&mut self) -> Option<&mut Neuron<T>> {
        if let Self::Neuron(neuron) = self {
            Some(neuron)
        } else {
            None
        }
    }
}

impl<T: Float> From<Bias<T>> for Gene<T> {
    fn from(bias: Bias<T>) -> Self {
        Self::Bias(bias)
    }
}

impl<T: Float> From<Input<T>> for Gene<T> {
    fn from(input: Input<T>) -> Self {
        Self::Input(input)
    }
}

impl<T: Float> From<Neuron<T>> for Gene<T> {
    fn from(neuron: Neuron<T>) -> Self {
        Self::Neuron(neuron)
    }
}

impl<T: Float> From<ForwardJumper<T>> for Gene<T> {
    fn from(forward: ForwardJumper<T>) -> Self {
        Self::ForwardJumper(forward)
    }
}

impl<T: Float> From<RecurrentJumper<T>> for Gene<T> {
    fn from(recurrent: RecurrentJumper<T>) -> Self {
        Self::RecurrentJumper(recurrent)
    }
}

/// Like [`Gene`], but cannot be a [`Neuron`] gene.
#[derive(Clone, Debug, PartialEq)]
pub enum NonNeuronGene<T> {
    /// See [`Bias`].
    Bias(Bias<T>),
    /// See [`Input`].
    Input(Input<T>),
    /// See [`ForwardJumper`].
    ForwardJumper(ForwardJumper<T>),
    /// See [`RecurrentJumper`].
    RecurrentJumper(RecurrentJumper<T>),
}

impl<T> From<Bias<T>> for NonNeuronGene<T> {
    fn from(bias: Bias<T>) -> Self {
        Self::Bias(bias)
    }
}

impl<T> From<Input<T>> for NonNeuronGene<T> {
    fn from(input: Input<T>) -> Self {
        Self::Input(input)
    }
}

impl<T> From<ForwardJumper<T>> for NonNeuronGene<T> {
    fn from(forward: ForwardJumper<T>) -> Self {
        Self::ForwardJumper(forward)
    }
}

impl<T> From<RecurrentJumper<T>> for NonNeuronGene<T> {
    fn from(recurrent: RecurrentJumper<T>) -> Self {
        Self::RecurrentJumper(recurrent)
    }
}

impl<T: Float> From<NonNeuronGene<T>> for Gene<T> {
    fn from(gene: NonNeuronGene<T>) -> Self {
        match gene {
            NonNeuronGene::Bias(bias) => Self::Bias(bias),
            NonNeuronGene::Input(input) => Self::Input(input),
            NonNeuronGene::ForwardJumper(forward) => Self::ForwardJumper(forward),
            NonNeuronGene::RecurrentJumper(recurrent) => Self::RecurrentJumper(recurrent),
        }
    }
}
