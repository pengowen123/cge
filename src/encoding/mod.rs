//! A portable encoding for `cge` [`Network`]s. See [`PortableCGE`].

mod error;
mod functions;
pub mod v1;

use serde::{Deserialize, Serialize};

use crate::network::{InvalidNetworkError, Network};

pub use error::Error;
pub(crate) use functions::*;
pub use v1::{Data, Metadata};

/// The portable encoding type, which can be serialized and deserialized to save or load networks
/// and their metadata.
///
/// [`Network::load_file`], [`Network::to_file`], and related methods are more convenient to use,
/// but this type must be used when deserializing from a format other than JSON. See
/// [`Network::to_serializable`] for serialization to different formats.
///
/// # Examples
///
/// ```
/// # let string =
/// #     include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/test_network_v1.cge"));
/// use cge::encoding::PortableCGE;
///
/// // Any format supported by `serde` can be used here
/// let deserialized: PortableCGE<()> = serde_json::from_str(&string).unwrap();
/// let (network, metadata, extra) = deserialized.build().unwrap();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "version", content = "network")]
pub enum PortableCGE<E> {
    #[serde(rename = "1")]
    V1(v1::Data<E>),
}

impl<E> PortableCGE<E> {
    /// Builds the `PortableCGE` into a [`Network`], its [`CommonMetadata`], and the user defined
    /// extra data.
    pub fn build(self) -> Result<(Network, CommonMetadata, E), InvalidNetworkError> {
        match self {
            Self::V1(e) => e.build(),
        }
    }
}

impl<E> From<v1::Data<E>> for PortableCGE<E> {
    fn from(net: Data<E>) -> Self {
        Self::V1(net)
    }
}

/// A metadata type that can be produced from all encoding versions.
///
/// `CommonMetadata` can also be converted to specific metadata versions if all required fields are
/// present using `Self::into_*`.
#[derive(Clone, Debug)]
pub struct CommonMetadata {
    pub description: Option<String>,
}

impl CommonMetadata {
    fn new(description: Option<String>) -> Self {
        Self { description }
    }

    /// Tries to convert `self` into the metadata type of the latest version, discarding any
    /// unused fields and returning `None` if any required fields are missing.
    pub fn into_latest_version(self) -> Option<Metadata> {
        Some(Metadata::new(self.description))
    }

    /// Converts `self` into the metadata type of version one, discarding any unused fields.
    pub fn into_v1(self) -> v1::Metadata {
        v1::Metadata::new(self.description)
    }
}

/// A trait implemented by all versioned encoding types.
pub trait EncodingVersion<E>: Into<PortableCGE<E>> {
    /// The metadata type for this version.
    type Metadata: Into<CommonMetadata>;

    /// Creates a [`PortableCGE`] from a network, its metadata, and the user-defined extra data,
    /// which can be set to `()` if unused.
    fn new(network: &Network, metadata: Self::Metadata, extra: E) -> PortableCGE<E>;

    /// Converts `self` into a [`Network`][crate::Network], its metadata, and the user-defined
    /// extra data.
    fn build(self) -> Result<(Network, CommonMetadata, E), InvalidNetworkError>;
}

/// A trait implemented by all versioned metadata types.
pub trait MetadataVersion<E> {
    type Data: EncodingVersion<E, Metadata = Self>;
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use super::*;

    fn get_file_path(file_name: &str) -> String {
        format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), file_name)
    }

    #[test]
    fn test_v1() {
        let mut loaded_string = String::new();

        let mut file = File::open(get_file_path("test_network_v1.cge")).unwrap();
        file.read_to_string(&mut loaded_string).unwrap();

        // Load and save a network in the v1 format
        let (network, metadata, extra) = Network::load_str::<()>(&loaded_string).unwrap();
        let metadata = v1::Metadata::new(metadata.description);
        let saved_string = network.to_string(metadata, extra).unwrap();

        assert_eq!(loaded_string.trim(), saved_string.trim());
    }
}
