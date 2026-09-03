/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared wire-format constants and validation helpers.

use crate::error::EncodedLabelIndexError;
use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, Write},
};

pub(crate) const LABEL_INDEX_MAGIC: [u8; 8] = *b"DANLBL01";
pub(crate) const LABEL_INDEX_VERSION: u32 = 1;
pub(crate) const BITSLICE_FORMAT: u32 = 0;
pub(crate) const MAX_LABEL_COUNT: usize = 1_000_000;
pub(crate) const MAX_LABEL_LENGTH: usize = 1 << 20;

pub(crate) fn validate_label(label: &str) -> Result<(), EncodedLabelIndexError> {
    if label.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot be empty".to_string(),
        ));
    }
    if label.contains(['&', '|']) {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label '{label}' contains a reserved filter delimiter"
        )));
    }
    if label.as_bytes().contains(&0) {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot contain embedded NUL bytes".to_string(),
        ));
    }
    if label.len() > MAX_LABEL_LENGTH {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label length {} exceeds limit {MAX_LABEL_LENGTH}",
            label.len()
        )));
    }
    if label.trim() != label {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot have leading or trailing whitespace".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_bitslice_padding(
    bits: &[u64],
    num_labels: usize,
    words_per_label: usize,
    num_vectors: u32,
) -> Result<(), EncodedLabelIndexError> {
    let remainder = (num_vectors as usize) % 64;
    if remainder == 0 || words_per_label == 0 {
        return Ok(());
    }

    let valid_mask = (1u64 << remainder) - 1;
    for label_id in 0..num_labels {
        let last_word = bits[(label_id + 1) * words_per_label - 1];
        if last_word & !valid_mask != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "bitslice padding contains out-of-range vector IDs".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

pub(crate) fn ensure_remaining(
    reader: &mut BufReader<File>,
    file_len: u64,
    required: usize,
    context: &str,
) -> Result<(), EncodedLabelIndexError> {
    let position = reader.stream_position()?;
    let remaining = file_len.checked_sub(position).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("reader position exceeds file length".to_string())
    })?;
    if required as u64 > remaining {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "{context} requires {required} bytes but only {remaining} remain"
        )));
    }
    Ok(())
}
