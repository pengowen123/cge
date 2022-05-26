#[cfg(feature = "serde_json")]
use serde_json;

use std::{error, fmt, io};

use crate::network;

/// An error while loading or saving a [`Network`][crate::Network] from/to the encoding.
#[derive(Debug)]
pub enum Error {
    /// An error during serialization or deserialization.
    #[cfg(feature = "serde_json")]
    Serde(serde_json::Error),
    /// An error while reading from/writing to a file.
    Io(io::Error),
    /// An error while constructing a [`Network`][crate::Network].
    CGE(network::Error),
    /// An error while setting the recurrest state of a [`Network`][crate::Network].
    RecurrentState(network::MismatchedLengthsError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "serde_json")]
            Self::Serde(e) => write!(f, "de/serialization error: {}", e),
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::CGE(e) => write!(f, "network error: {}", e),
            Self::RecurrentState(e) => write!(f, "failed to set recurrent state: {}", e),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "serde_json")]
            Self::Serde(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::CGE(e) => Some(e),
            Self::RecurrentState(e) => Some(e),
        }
    }
}

#[cfg(feature = "serde_json")]
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<network::Error> for Error {
    fn from(e: network::Error) -> Self {
        Self::CGE(e)
    }
}

impl From<network::MismatchedLengthsError> for Error {
    fn from(e: network::MismatchedLengthsError) -> Self {
        Self::RecurrentState(e)
    }
}
