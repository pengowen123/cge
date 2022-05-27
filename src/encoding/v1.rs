//! Version one of the encoding. See [`Data`] and [`Metadata`].

use num_traits::Float;
use serde::{Deserialize, Serialize};

use super::{
    CommonMetadata, EncodingVersion, Extra, MetadataVersion, PortableCGE, WithRecurrentState,
};
use crate::activation::Activation;
use crate::gene::Gene;
use crate::Network;

/// A type for encoding a [`Network`][crate::Network] and its metadata in version one of the
/// format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Data<T: Float, E> {
    /// Metadata for the [`Network`].
    pub metadata: Metadata,
    /// The activation function of the [`Network`].
    pub activation: Activation,
    /// The genome of the [`Network`].
    pub genome: Vec<Gene<T>>,
    /// The recurrent state of the [`Network`][crate::Network] (see
    /// [`Network::recurrent_state`][crate::Network::recurrent_state]).
    pub recurrent_state: Option<Vec<T>>,
    /// Arbitrary, user-defined extra data for the [`Network`].
    pub extra: Extra<E>,
}

impl<T: Float, E> EncodingVersion<T, E> for Data<T, E> {
    type Metadata = Metadata;

    fn new(
        network: &Network<T>,
        metadata: Self::Metadata,
        extra: E,
        with_state: WithRecurrentState,
    ) -> PortableCGE<T, E> {
        let recurrent_state = if with_state.0 {
            Some(network.recurrent_state().collect())
        } else {
            None
        };
        Self {
            metadata,
            activation: network.activation(),
            genome: network.genome().into(),
            recurrent_state,
            extra: Extra::Ok(extra),
        }
        .into()
    }

    fn build(
        self,
        with_state: WithRecurrentState,
    ) -> Result<(Network<T>, CommonMetadata, Extra<E>), super::Error> {
        let mut network = Network::new(self.genome, self.activation)?;

        if with_state.0 {
            if let Some(state) = self.recurrent_state {
                network.set_recurrent_state(&state)?;
            }
        }

        Ok((network, self.metadata.into(), self.extra))
    }
}

/// Version one metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    /// An optional description for a [`Network`].
    pub description: Option<String>,
}

impl Metadata {
    /// Returns a new `Metadata` for a [`Network`].
    pub fn new(description: Option<String>) -> Self {
        Self { description }
    }
}

impl From<Metadata> for CommonMetadata {
    fn from(m: Metadata) -> CommonMetadata {
        CommonMetadata::new(m.description)
    }
}

impl<T: Float, E> MetadataVersion<T, E> for Metadata {
    type Data = Data<T, E>;
}
