//! A portable encoding for `cge` [`Network`]s. See [`PortableCGE`].

mod error;
#[cfg(feature = "serde_json")]
mod functions;
pub mod v1;

use serde::{Deserialize, Serialize};

use crate::Network;

pub use error::Error;
#[cfg(feature = "serde_json")]
pub(crate) use functions::*;

/// The latest metadata version.
pub type Metadata = v1::Metadata;
/// The latest encoding version.
pub type Data<E> = v1::Data<E>;

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
/// use cge::encoding::{PortableCGE, WithRecurrentState};
///
/// // Any format supported by `serde` can be used here
/// let deserialized: PortableCGE<()> = serde_json::from_str(&string).unwrap();
/// let (network, metadata, extra) = deserialized.build(WithRecurrentState(true)).unwrap();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "version", content = "network")]
pub enum PortableCGE<E> {
    #[serde(rename = "1")]
    V1(v1::Data<E>),
}

impl<E> PortableCGE<E> {
    /// Builds the `PortableCGE` into a [`Network`], its [`CommonMetadata`], and the user defined
    /// extra data. If `with_state` is `true`, the [`Network`]'s recurrent state is loaded if it
    /// exists. If not loaded, it is initialized to all zeroes.
    pub fn build(
        self,
        with_state: WithRecurrentState,
    ) -> Result<(Network, CommonMetadata, E), Error> {
        match self {
            Self::V1(e) => e.build(with_state),
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

/// Whether to save or load the recurrent state of a [`Network`].
#[derive(Clone, Copy, Debug)]
pub struct WithRecurrentState(pub bool);

/// A trait implemented by all versioned encoding types.
pub trait EncodingVersion<E>: Into<PortableCGE<E>> {
    /// The metadata type for this version.
    type Metadata: Into<CommonMetadata>;

    /// Creates a [`PortableCGE`] from a network, its metadata, and the user-defined extra data,
    /// which can be set to `()` if unused. If `with_state` is `true`, the [`Network`]'s recurrent
    /// state is loaded if it exists. If not loaded, it is initialized to all zeroes.
    #[allow(clippy::new_ret_no_self)]
    fn new(
        network: &Network,
        metadata: Self::Metadata,
        extra: E,
        with_state: WithRecurrentState,
    ) -> PortableCGE<E>;

    /// Converts `self` into a [`Network`][crate::Network], its metadata, and the user-defined
    /// extra data. The network's recurrent state is loaded if it exists and `with_state` is `true`.
    fn build(self, with_state: WithRecurrentState) -> Result<(Network, CommonMetadata, E), Error>;
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
        let (network, metadata, extra) =
            Network::load_str::<()>(&loaded_string, WithRecurrentState(true)).unwrap();
        let metadata = v1::Metadata::new(metadata.description);
        let saved_string = network
            .to_string(metadata, extra, WithRecurrentState(true))
            .unwrap();
        assert_eq!(loaded_string.trim(), saved_string.trim());
    }
}
