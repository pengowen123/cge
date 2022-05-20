//! Version one of the encoding.

use serde::{Deserialize, Serialize};

use super::{CommonMetadata, EncodingVersion, MetadataVersion, PortableCGE};
use crate::activation::Activation;
use crate::gene::Gene;
use crate::network::{self, Network};

/// A type for encoding a [`Network`][crate::Network] and its metadata in version one of the
/// format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Data<E> {
    pub metadata: Metadata,
    pub activation: Activation,
    pub genome: Vec<Gene>,
    pub extra: E,
}

impl<E> EncodingVersion<E> for Data<E> {
    type Metadata = Metadata;

    fn new(network: &Network, metadata: Self::Metadata, extra: E) -> PortableCGE<E> {
        Self {
            metadata,
            genome: network.genome().into(),
            activation: network.activation(),
            extra,
        }
        .into()
    }

    fn build(self) -> Result<(Network, CommonMetadata, E), network::Error> {
        let network = Network::new(self.genome, self.activation)?;

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
