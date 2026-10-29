/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::IntoANNResult, neighbor::Neighbor, utils::IntoUsize};

use crate::{
    num::IdLimit,
    prefetch::{self, Prefetch},
    repr, store,
};

use super::{OutOfBounds, RawDistance, RawQueryDistance};

#[derive(Debug)]
pub(in crate::repr) struct ExpandBeam<'a, D, P> {
    reader: store::sidecar::Reader<'a>,
    distance: D,
    prefetch: prefetch::Checked<P>,
    lookahead: Option<NonZeroUsize>,
    bitmap_lookahead: Option<NonZeroUsize>,
}

impl<'a, D, P: Prefetch> ExpandBeam<'a, D, P> {
    pub(in crate::repr) fn new(
        reader: store::sidecar::Reader<'a>,
        distance: D,
        prefetch: P,
        lookahead: Option<NonZeroUsize>,
        bitmap_lookahead: Option<NonZeroUsize>,
    ) -> Self {
        let prefetch = prefetch::Checked::new(prefetch, reader.bytes())
            .expect("internal APIs should only provide valid prefetchers");
        Self {
            reader,
            distance,
            prefetch,
            lookahead,
            bitmap_lookahead,
        }
    }

    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    /// # Safety
    ///
    /// `i` must be in bounds for the reader.
    #[inline(always)]
    unsafe fn prefetch_data(&self, i: u32) {
        // SAFETY: The caller guarantees bounds. Raw access does not require
        // readability, and the prefetcher was validated against payload bytes.
        unsafe {
            let raw = self.reader.read_raw_unchecked(i.into_usize());
            self.prefetch.prefetch(raw.as_ptr(), raw.len());
        }
    }
}

// SAFETY: id_limit and expand_beam use the same reader bounds.
unsafe impl<D: RawQueryDistance, P: Prefetch> repr::ExpandBeam for ExpandBeam<'_, D, P> {
    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i.into_usize()) {
            return Err(ANNError::new(OutOfBounds::new(i)));
        }
        // SAFETY: Bounds checked above.
        match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
            Some(data) => Ok(Some(self.distance.eval(data).into_ann_result()?)),
            None => Ok(None),
        }
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [Neighbor<u32>]) -> ANNResult<usize> {
        debug_assert!(buffer.len() >= list.len());
        let len = list.len();
        let bitmap = self.bitmap_lookahead.map_or(0, NonZeroUsize::get).min(len);
        let data = self.lookahead.map_or(0, NonZeroUsize::get).min(len);

        for &id in list.iter().take(bitmap) {
            // SAFETY: The caller guarantees all IDs are in bounds.
            unsafe { self.reader.prefetch_bitmap(id.into_usize()) };
        }
        for &id in list.iter().take(data) {
            // SAFETY: The caller guarantees all IDs are in bounds.
            unsafe { self.prefetch_data(id) };
        }

        let mut b = if bitmap == 0 { len } else { bitmap };
        let mut d = if data == 0 { len } else { data };
        let mut processed = 0;
        for &id in list {
            if b != len {
                // SAFETY: b is in bounds and all IDs are valid for the reader.
                unsafe { self.reader.prefetch_bitmap(list.get_unchecked(b).into_usize()) };
                b += 1;
            }
            if d != len {
                // SAFETY: d is in bounds and all IDs are valid for the reader.
                unsafe { self.prefetch_data(*list.get_unchecked(d)) };
                d += 1;
            }
            // SAFETY: The caller guarantees bounds. The reader performs the
            // acquire check before materializing any reference to the payload.
            if let Some(payload) = unsafe { self.reader.read_in_bounds(id.into_usize()) } {
                let distance = self.distance.eval(payload).into_ann_result()?;
                // SAFETY: At most one output per input; caller guarantees capacity.
                *unsafe { buffer.get_unchecked_mut(processed) } = Neighbor::new(id, distance);
                processed += 1;
            }
        }
        Ok(processed)
    }
}

#[derive(Debug)]
pub(in crate::repr) struct Prune<'a, D> {
    reader: store::sidecar::Reader<'a>,
    distance: D,
}

impl<'a, D> Prune<'a, D> {
    pub(in crate::repr) fn new(reader: store::sidecar::Reader<'a>, distance: D) -> Self {
        Self { reader, distance }
    }

    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

impl<D: RawDistance> repr::Prune for Prune<'_, D> {
    fn get(&self, id: u32) -> Option<&[u8]> {
        self.reader.read(id.into_usize())
    }

    fn evaluate(&self, a: &[u8], b: &[u8]) -> f32 {
        self.distance
            .eval(a, b)
            .expect("diskann currently does not support fallible prune")
    }
}
