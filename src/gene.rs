//! Different types of genes that can be used in a network genome.

/// A bias gene.
///
/// Adds a constant value to the network.
#[derive(Clone, Debug)]
pub struct Bias {
    value: f64,
}

impl Bias {
    /// Returns a new `Bias` that adds a constant `value` to the network.
    pub fn new(value: f64) -> Self {
        Self {
            value,
        }
    }

    /// Returns the value of the `Bias`.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// The ID of a network input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug)]
pub struct Input {
    // The ID of the network input referred to
    id: InputId,
    weight: f64,
}

impl Input {
    /// Returns a new `Input` that connects to the network input with the id and weights it by
    /// `weight`.
    pub fn new(id: InputId, weight: f64) -> Self {
        Self {
            id,
            weight,
        }
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug)]
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
    current_value: Option<f64>,
    // The unweighted value outputted by this neuron during the previous network evaluation
    previous_value: f64,
}

impl Neuron {
    /// Returns a new `Neuron` that takes `num_inputs` inputs and weights its output by `weight`.
    ///
    /// If specifying the neuron id is unnecessary (i.e., when adding a new one to a network),
    /// [`without_id`][Self::without_id] can be used instead.
    pub fn new(id: NeuronId, num_inputs: usize, weight: f64,) -> Self {
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
#[derive(Clone, Debug)]
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
        Self {
            source_id,
            weight,
        }
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
#[derive(Clone, Debug)]
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
        Self {
            source_id,
            weight,
        }
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
#[derive(Clone, Debug)]
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
    fn from(forward_jumper: ForwardJumper) -> Self {
        Self::ForwardJumper(forward_jumper)
    }
}

impl From<RecurrentJumper> for Gene {
    fn from(recurrent_jumper: RecurrentJumper) -> Self {
        Self::RecurrentJumper(recurrent_jumper)
    }
}
