//! Version one of the encoding.

use serde::{Deserialize, Serialize};

use super::{CommonMetadata, EncodingVersion, MetadataVersion, PortableCGE, WithRecurrentState};
use crate::activation::Activation;
use crate::gene::Gene;
use crate::Network;

/// A type for encoding a [`Network`][crate::Network] and its metadata in version one of the
/// format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Data<E> {
    pub metadata: Metadata,
    pub activation: Activation,
    pub genome: Vec<Gene>,
    pub recurrent_state: Option<Vec<f64>>,
    pub extra: E,
}

impl<E> EncodingVersion<E> for Data<E> {
    type Metadata = Metadata;

    fn new(
        network: &Network,
        metadata: Self::Metadata,
        extra: E,
        with_state: WithRecurrentState,
    ) -> PortableCGE<E> {
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
            extra,
        }
        .into()
    }

    fn build(
        self,
        with_state: WithRecurrentState,
    ) -> Result<(Network, CommonMetadata, E), super::Error> {
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
    pub description: Option<String>,
}

impl Metadata {
    pub fn new<S: Into<Option<String>>>(description: S) -> Self {
        Self {
            description: description.into(),
        }
    }
}

impl From<Metadata> for CommonMetadata {
    fn from(m: Metadata) -> CommonMetadata {
        CommonMetadata::new(m.description)
    }
}

impl<E> MetadataVersion<E> for Metadata {
    type Data = Data<E>;
}
