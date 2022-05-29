//! A portable encoding for `cge` [`Network`]s. See [`PortableCGE`].

mod error;
#[cfg(feature = "json")]
mod functions;
pub mod v1;

use num_traits::Float;
use serde::{Deserialize, Deserializer, Serialize};

use crate::Network;

pub use error::Error;
#[cfg(feature = "json")]
pub(crate) use functions::*;

/// The latest metadata version.
pub type Metadata = v1::Metadata;
/// The latest encoding version.
pub type Data<T, E> = v1::Data<T, E>;

/// The portable encoding type, which can be serialized and deserialized to save and load
/// [`Network`]s and their metadata.
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
/// // `()` is the user-defined extra data type
/// let deserialized: PortableCGE<f64, ()> = serde_json::from_str(&string).unwrap();
/// let (network, metadata, extra) = deserialized.build(WithRecurrentState(true)).unwrap();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "version", content = "network")]
pub enum PortableCGE<T: Float, E> {
    /// Version one of the encoding.
    #[serde(rename = "1")]
    V1(v1::Data<T, E>),
}

impl<T: Float, E> PortableCGE<T, E> {
    /// Builds the `PortableCGE` into a [`Network`], its [`CommonMetadata`], and the user defined
    /// extra data. If `with_state` is `true`, the [`Network`]'s recurrent state is loaded if it
    /// exists. If not loaded, it is initialized to all zeroes.
    ///
    /// The extra data returned will be [`Extra::Ok`] if it matches the requested type `E`, or
    /// [`Extra::Other`] otherwise.
    pub fn build(
        self,
        with_state: WithRecurrentState,
    ) -> Result<(Network<T>, CommonMetadata, Extra<E>), Error> {
        match self {
            Self::V1(e) => e.build(with_state),
        }
    }
}

impl<T: Float, E> From<v1::Data<T, E>> for PortableCGE<T, E> {
    fn from(net: Data<T, E>) -> Self {
        Self::V1(net)
    }
}

/// A metadata type that can be produced from all encoding versions.
///
/// `CommonMetadata` can also be converted to specific metadata versions if all required fields are
/// present using `Self::into_*`.
#[derive(Clone, Debug)]
pub struct CommonMetadata {
    /// An optional description for a [`Network`].
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
pub trait EncodingVersion<T: Float, E>: Into<PortableCGE<T, E>> {
    /// The metadata type for this version.
    type Metadata: Into<CommonMetadata>;

    /// Creates a [`PortableCGE`] from a [`Network`], its metadata, and the user-defined extra data,
    /// which can be set to `()` if unused. If `with_state` is `true`, the [`Network`]'s recurrent
    /// state is loaded if it exists. If not loaded, it is initialized to all zeroes.
    #[allow(clippy::new_ret_no_self)]
    fn new(
        network: &Network<T>,
        metadata: Self::Metadata,
        extra: E,
        with_state: WithRecurrentState,
    ) -> PortableCGE<T, E>;

    /// Converts `self` into a [`Network`][crate::Network], its metadata, and the user-defined
    /// extra data. The network's recurrent state is loaded if it exists and `with_state` is `true`.
    fn build(
        self,
        with_state: WithRecurrentState,
    ) -> Result<(Network<T>, CommonMetadata, Extra<E>), Error>;
}

/// A trait implemented by all versioned metadata types.
pub trait MetadataVersion<T: Float, E>: Into<CommonMetadata> + Sized {
    /// The data type corresponding to this metadata's encoding version.
    type Data: EncodingVersion<T, E, Metadata = Self>;
}

/// User-defined extra data to be serialized or deserialized alongside a [`Network`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Extra<E> {
    /// The data deserialized into the requested type successfully.
    Ok(E),
    /// A different type than requested was encountered.
    #[serde(deserialize_with = "deserialize_ignore_any")]
    Other,
}

impl<E> Extra<E> {
    /// Returns the contained data if it exists.
    pub fn unwrap(self) -> E {
        if let Self::Ok(data) = self {
            data
        } else {
            panic!("called `unwrap` on an `Other` value");
        }
    }

    /// Returns whether `self` is an `Ok` value.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Returns whether `self` is an `Other` value.
    pub fn is_other(&self) -> bool {
        matches!(self, Self::Other)
    }
}

/// Deserializes any value into `T::default`.
fn deserialize_ignore_any<'de, D: Deserializer<'de>, T: Default>(
    deserializer: D,
) -> Result<T, D::Error> {
    serde::de::IgnoredAny::deserialize(deserializer).map(|_| T::default())
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
    fn test_extra() {
        #[derive(Serialize, Deserialize)]
        struct Foo {
            x: i32,
            y: [f64; 2],
        }

        let path = get_file_path("with_extra_data_v1.cge");
        let (_, _, extra) =
            Network::<f64>::load_file::<(), _>(&path, WithRecurrentState(false)).unwrap();

        // The stored extra data is not `()`
        assert!(extra.is_other());

        let (_, _, extra2) =
            Network::<f64>::load_file::<Foo, _>(&path, WithRecurrentState(false)).unwrap();

        // The stored extra data is `Foo`
        assert!(extra2.is_ok());
    }

    #[test]
    fn test_v1() {
        let mut loaded_string = String::new();

        let mut file = File::open(get_file_path("test_network_v1.cge")).unwrap();
        file.read_to_string(&mut loaded_string).unwrap();

        // Load and save a network in the v1 format
        let (mut network, metadata, extra) =
            Network::<f64>::load_str::<()>(&loaded_string, WithRecurrentState(true)).unwrap();

        let metadata = metadata.into_v1();
        let saved_string = network
            .to_string(metadata, extra, WithRecurrentState(true))
            .unwrap();
        assert_eq!(loaded_string.trim(), saved_string.trim());

        // Check that recurrent state is loaded properly
        let (mut network2, _, _) =
            Network::<f64>::load_str::<()>(&loaded_string, WithRecurrentState(false)).unwrap();
        let inputs = &[1.0, 1.0];
        assert_ne!(network.evaluate(inputs), network2.evaluate(inputs));
    }
}
