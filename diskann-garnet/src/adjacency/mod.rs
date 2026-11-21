/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Garnet record framing and Tinyid v1 sorted adjacency payloads.

mod adaptive;
mod bits;
mod rice;
mod stream;
#[cfg(test)]
mod tests;
#[cfg(target_arch = "x86_64")]
mod x86;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum AdjacencyCodec {
    #[default]
    Identity = 0,
    StreamVByte64 = 1,
    AdaptiveGap = 2,
    Rice = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CodecError {
    #[error("unknown adjacency codec {0}")]
    UnknownCodec(u8),
    #[error("adjacency degree exceeds the configured maximum")]
    InvalidDegree,
    #[error("adjacency output buffer is too small")]
    OutputTooSmall,
    #[error("adjacency width scratch is too small")]
    WidthScratchTooSmall,
    #[error("adjacency size exceeds the addressable range")]
    SizeOverflow,
    #[error("malformed adjacency record: {0}")]
    Malformed(&'static str),
}

impl TryFrom<u8> for AdjacencyCodec {
    type Error = CodecError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Identity),
            1 => Ok(Self::StreamVByte64),
            2 => Ok(Self::AdaptiveGap),
            3 => Ok(Self::Rice),
            _ => Err(CodecError::UnknownCodec(value)),
        }
    }
}

/// Sorted compressed list payload, excluding the common count and record framing.
trait ListCodec {
    fn max_payload_len(max_degree: usize) -> usize;
    fn encode(ids: &[u64], out: &mut [u8], widths: &mut [u8]) -> Result<usize, CodecError>;
    fn decode(payload: &[u8], out: &mut [u64]) -> Result<(), CodecError>;
}

impl AdjacencyCodec {
    pub(crate) fn max_encoded_len(self, max_degree: usize) -> usize {
        let payload = match self {
            Self::Identity => return identity_len(max_degree).unwrap_or(usize::MAX),
            Self::StreamVByte64 => stream::Stream::max_payload_len(max_degree),
            Self::AdaptiveGap => adaptive::Adaptive::max_payload_len(max_degree),
            Self::Rice => rice::Rice::max_payload_len(max_degree),
        };
        // Codec byte, count, and a possible suffix to avoid the identity length.
        payload.saturating_add(2 + uleb_len(max_degree as u64))
    }

    /// Encode into caller storage, sorting only compressed lists. AdaptiveGap
    /// uses one scratch byte per ID. On error the scratch and output may change.
    pub(crate) fn encode(
        self,
        ids: &mut [u64],
        max_degree: usize,
        out: &mut [u8],
        widths: &mut [u8],
    ) -> Result<usize, CodecError> {
        if ids.len() > max_degree {
            return Err(CodecError::InvalidDegree);
        }
        let identity_size = identity_len(max_degree)?;
        if self == Self::Identity {
            let out = out
                .get_mut(..identity_size)
                .ok_or(CodecError::OutputTooSmall)?;
            let (slots, count) = out.split_at_mut(identity_size - 8);
            let (values, padding) = slots.split_at_mut(ids.len() * 8);
            values.copy_from_slice(bytemuck::cast_slice(ids));
            padding.fill(0);
            count.copy_from_slice(&(ids.len() as u64).to_ne_bytes());
            return Ok(identity_size);
        }

        let mut cursor = 0;
        put_byte(out, &mut cursor, self as u8)?;
        put_uleb(out, &mut cursor, ids.len() as u64)?;
        if !ids.is_sorted() {
            ids.sort_unstable();
        }
        let payload = &mut out[cursor..];
        cursor += match self {
            Self::Identity => unreachable!(),
            Self::StreamVByte64 => stream::Stream::encode(ids, payload, widths)?,
            Self::AdaptiveGap => adaptive::Adaptive::encode(ids, payload, widths)?,
            Self::Rice => rice::Rice::encode(ids, payload, widths)?,
        };
        if cursor >= identity_size {
            put_byte(out, &mut cursor, 0)?;
        }
        Ok(cursor)
    }

    /// Detect the stored codec independently of the writer's configuration.
    /// Historical identity records are distinguished by length, not a tag, and
    /// their unused slots may contain stale IDs. Output is unspecified on error.
    pub(crate) fn decode(
        record: &[u8],
        max_degree: usize,
        out: &mut [u64],
    ) -> Result<usize, CodecError> {
        let identity_size = identity_len(max_degree)?;
        if record.len() == identity_size {
            let (slots, count) = record
                .split_last_chunk::<8>()
                .expect("identity is at least 8 bytes");
            let count = checked_count(u64::from_ne_bytes(*count), max_degree)?;
            let out = out.get_mut(..count).ok_or(CodecError::OutputTooSmall)?;
            for (id, slot) in out.iter_mut().zip(slots.as_chunks::<8>().0) {
                *id = u64::from_ne_bytes(*slot);
            }
            return Ok(count);
        }
        let mut record = if record.len() > identity_size {
            record
                .strip_suffix(&[0])
                .ok_or(CodecError::Malformed("invalid size suffix"))?
        } else {
            record
        };
        let codec = Self::try_from(take_byte(&mut record)?)?;
        if codec == Self::Identity {
            return Err(CodecError::Malformed("identity has no codec tag"));
        }
        let count = checked_count(take_uleb(&mut record)?, max_degree)?;
        let out = out.get_mut(..count).ok_or(CodecError::OutputTooSmall)?;
        match codec {
            Self::Identity => unreachable!(),
            Self::StreamVByte64 => stream::Stream::decode(record, out)?,
            Self::AdaptiveGap => adaptive::Adaptive::decode(record, out)?,
            Self::Rice => rice::Rice::decode(record, out)?,
        }
        Ok(count)
    }
}

fn identity_len(max_degree: usize) -> Result<usize, CodecError> {
    max_degree
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or(CodecError::SizeOverflow)
}

fn checked_count(count: u64, max_degree: usize) -> Result<usize, CodecError> {
    usize::try_from(count)
        .ok()
        .filter(|n| *n <= max_degree)
        .ok_or(CodecError::InvalidDegree)
}

fn uleb_len(value: u64) -> usize {
    (64 - value.leading_zeros()).max(1).div_ceil(7) as usize
}

fn put_byte(out: &mut [u8], cursor: &mut usize, byte: u8) -> Result<(), CodecError> {
    *out.get_mut(*cursor).ok_or(CodecError::OutputTooSmall)? = byte;
    *cursor += 1;
    Ok(())
}

fn put_uleb(out: &mut [u8], cursor: &mut usize, mut value: u64) -> Result<(), CodecError> {
    while value >= 128 {
        put_byte(out, cursor, (value as u8) | 0x80)?;
        value >>= 7;
    }
    put_byte(out, cursor, value as u8)
}

fn take_byte(payload: &mut &[u8]) -> Result<u8, CodecError> {
    let (&byte, rest) = payload
        .split_first()
        .ok_or(CodecError::Malformed("truncated header"))?;
    *payload = rest;
    Ok(byte)
}

fn take_uleb(payload: &mut &[u8]) -> Result<u64, CodecError> {
    let mut value = 0;
    for i in 0..10 {
        let byte = take_byte(payload)?;
        if i == 9 && byte > 1 {
            return Err(CodecError::Malformed("ULEB128 exceeds u64"));
        }
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            if i != 0 && byte == 0 {
                return Err(CodecError::Malformed("noncanonical ULEB128"));
            }
            return Ok(value);
        }
    }
    unreachable!("the tenth ULEB128 byte always terminates")
}

fn take<'a>(payload: &mut &'a [u8], n: usize) -> Result<&'a [u8], CodecError> {
    let (head, tail) = payload
        .split_at_checked(n)
        .ok_or(CodecError::Malformed("truncated payload"))?;
    *payload = tail;
    Ok(head)
}

fn finished(payload: &[u8]) -> Result<(), CodecError> {
    if payload.is_empty() {
        Ok(())
    } else {
        Err(CodecError::Malformed("trailing payload bytes"))
    }
}

fn add_gap(previous: &mut u64, gap: u64, id: &mut u64) -> Result<(), CodecError> {
    *previous = previous
        .checked_add(gap)
        .ok_or(CodecError::Malformed("adjacency ID overflows u64"))?;
    *id = *previous;
    Ok(())
}
