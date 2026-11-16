/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use std::{error::Error, fs, io};

pub fn load(path: &str) -> Result<rowmajor::Owned<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(io::Error::other("missing fbin header").into());
    }
    let n = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let expected = n
        .checked_mul(d)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(8));
    if n == 0 || d == 0 || expected != Some(bytes.len()) {
        return Err(io::Error::other("invalid fbin dimensions or length").into());
    }
    let mut result = rowmajor::Owned::try_from_element(n, d, 0.0)?;
    for (out, bytes) in result
        .as_mut_slice()
        .iter_mut()
        .zip(bytes[8..].chunks_exact(4))
    {
        *out = f32::from_le_bytes(bytes.try_into()?);
        if !out.is_finite() {
            return Err(io::Error::other("non-finite input").into());
        }
    }
    Ok(result)
}

pub fn prefix(
    data: rowmajor::Ref<'_, f32>,
    n: usize,
) -> Result<rowmajor::Ref<'_, f32>, Box<dyn Error>> {
    if n == 0 || n > data.nrows() {
        return Err(io::Error::other("invalid row count").into());
    }
    let d = data.ncols();
    Ok(
        rowmajor::Ref::try_from_data(&data.into_slice()[..n * d], n, d)
            .map_err(|_| io::Error::other("invalid prefix layout"))?,
    )
}
