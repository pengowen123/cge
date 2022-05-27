//! Functions that wrap up the encoding functionality.

use num_traits::Float;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use std::fs::{DirBuilder, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use super::{CommonMetadata, Error, Extra, PortableCGE, WithRecurrentState};
use crate::network::Network;

/// Loads encoded data of any version from a string. Also loads the network's recurrent state if
/// `with_state` is `true`.
pub(crate) fn load_str<'a, T, E>(
    s: &'a str,
    with_state: WithRecurrentState,
) -> Result<(Network<T>, CommonMetadata, Extra<E>), Error>
where
    T: Deserialize<'a> + Float,
    E: Deserialize<'a>,
{
    serde_json::from_str::<PortableCGE<T, E>>(s)?
        .build(with_state)
        .map_err(Into::into)
}

/// Loads encoded data of any version from a file. Also loads the network's recurrent state if
/// `with_state` is `true`.
pub(crate) fn load_file<T, E, P>(
    path: P,
    with_state: WithRecurrentState,
) -> Result<(Network<T>, CommonMetadata, Extra<E>), Error>
where
    T: DeserializeOwned + Float,
    E: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    serde_json::from_reader::<_, PortableCGE<T, E>>(reader)?
        .build(with_state)
        .map_err(Into::into)
}

/// Encodes data in a specific encoding version to a string.
pub(crate) fn to_string<T, E>(serializable: PortableCGE<T, E>) -> Result<String, Error>
where
    T: Serialize + Float,
    E: Serialize,
{
    serde_json::to_string_pretty(&serializable).map_err(Into::into)
}

/// Encodes data in a specific encoding version to a file.
///
/// Recursively creates missing directories if `create_dirs` is `true`.
pub(crate) fn to_file<T, E, P>(
    serializable: PortableCGE<T, E>,
    path: P,
    create_dirs: bool,
) -> Result<(), Error>
where
    T: Serialize + Float,
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
    let result = serde_json::to_writer_pretty(&mut writer, &serializable)?;

    writer.flush()?;

    Ok(result)
}
