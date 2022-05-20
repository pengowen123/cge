use serde_json;

use std::{error, fmt, io};

use crate::network::InvalidNetworkError;

/// An error while loading or saving a [`Network`][crate::Network] from/to the encoding.
#[derive(Debug)]
pub enum Error {
    /// An error during serialization or deserialization.
    Serde(serde_json::Error),
    /// An error while reading from/writing to a file.
    Io(io::Error),
    /// An error while constructing a [`Network`][crate::Network].
    CGE(InvalidNetworkError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Serde(e) => write!(f, "de/serialization error: {}", e),
            Self::Io(e) => write!(f, "io error: {}", e),
            Self::CGE(e) => write!(f, "network error: {}", e),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Serde(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::CGE(e) => Some(e),
        }
    }
}

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

impl From<InvalidNetworkError> for Error {
    fn from(e: InvalidNetworkError) -> Self {
        Self::CGE(e)
    }
}
