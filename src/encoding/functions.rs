//! Functions that wrap up the encoding functionality.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::fs::{DirBuilder, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::{CommonMetadata, Error, PortableCGE};
use crate::network::Network;

/// Loads encoded data of any version from a string.
pub(crate) fn load_str<'a, E>(s: &'a str) -> Result<(Network, CommonMetadata, E), Error>
where
    E: Deserialize<'a>,
{
    serde_json::from_str::<PortableCGE<E>>(s)?
        .build()
        .map_err(Into::into)
}

/// Loads encoded data of any version from a file.
pub(crate) fn load_file<E, P>(path: P) -> Result<(Network, CommonMetadata, E), Error>
where
    E: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    serde_json::from_reader::<_, PortableCGE<E>>(reader)?
        .build()
        .map_err(Into::into)
}

/// Encodes data in a specific encoding version to a string.
pub(crate) fn to_string<E>(serializable: PortableCGE<E>) -> Result<String, Error>
where
    E: Serialize,
{
    serde_json::to_string_pretty(&serializable).map_err(Into::into)
}

/// Encodes data in a specific encoding version to a file.
///
/// Recursively creates missing directories if `create_dirs` is `true`.
pub(crate) fn to_file<E, P>(
    serializable: PortableCGE<E>,
    path: P,
    create_dirs: bool,
) -> Result<(), Error>
where
    E: Serialize,
    P: AsRef<Path>,
{
    let path = path.as_ref();

    if create_dirs {
        if let Some(parent) = path.parent() {
            DirBuilder::new().recursive(true).create(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let result =
        serde_json::to_writer_pretty(&mut writer, &serializable)?;

    writer.flush()?;

    Ok(result)
}
