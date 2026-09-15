/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::IntoANNResult, utils::IntoUsize, neighbor::Neighbor};

use crate::{
    num::IdLimit,
    prefetch::{self, Prefetch},
    repr, store,
};

use super::{OutOfBounds, RawQueryDistance};

////////////////
// ExpandBeam //
////////////////

#[derive(Debug)]
pub(in crate::repr) struct ExpandBeam<'a, D, P> {
    reader: store::intrusive::Reader<'a>,
    distance: D,
    prefetch: prefetch::Checked<P>,
    lookahead: Option<NonZeroUsize>,
}

impl<'a, D, P> ExpandBeam<'a, D, P> {
    // TODO: Should this be unsafe pending the relationship between `distance` and by
    // number of bytes in `reader`?
    pub(in crate::repr) fn new(
        reader: store::intrusive::Reader<'a>,
        distance: D,
        prefetch: P,
        lookahead: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Prefetch,
    {
        let prefetch = prefetch::Checked::new(prefetch, reader.bytes_plus_tag())
            .expect("internal APIs should only provide valid prefetchers");

        Self {
            reader,
            distance,
            prefetch,
            lookahead,
        }
    }

    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

unsafe impl<D, P> repr::ExpandBeam for ExpandBeam<'_, D, P>
where
    P: Prefetch,
    D: RawQueryDistance,
{
    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i.into_usize()) {
            Err(ANNError::new(OutOfBounds::new(i)))
        } else {
            // SAFETY: We have checked that `i` is in-bounds.
            match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                Some(data) => {
                    let distance = self.distance.eval(data).into_ann_result()?;
                    Ok(Some(distance))
                }
                None => Ok(None),
            }
        }
    }

    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [Neighbor<u32>]) -> ANNResult<usize> {
        debug_assert!(buffer.len() >= list.len());

        let len = list.len();
        let lookahead = self.lookahead.map(|l| l.get()).unwrap_or(0).min(len);

        for j in list.iter().take(lookahead) {
            // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as well
            // as the validity of the prefetch bounds.
            //
            // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon construction.
            //
            // We do not materialize the `RawSlice` as a reference.
            unsafe {
                let raw = self.reader.read_raw_unchecked(j.into_usize());
                self.prefetch.prefetch(raw.as_ptr(), raw.len());
            }
        }

        // Disable prefetching if the lookahead is 0.
        let mut j = if lookahead == 0 { len } else { lookahead };
        let mut processed = 0;
        for &i in list.iter() {
            if j != len {
                // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as
                // well as the validity of the prefetch bounds.
                //
                // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon
                // construction.
                //
                // We do not materialize the `RawSlice` as a reference.
                unsafe {
                    let raw = self
                        .reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize());
                    self.prefetch.prefetch(raw.as_ptr(), raw.len());
                }
                j += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                let distance = self.distance.eval(data).into_ann_result()?;

                // SAFETY: Inherited from caller.
                *unsafe { buffer.get_unchecked_mut(processed) } = Neighbor::new(i, distance);
                processed += 1;
            }
        }

        Ok(processed)
    }
}

///////////
// Prune //
///////////

#[derive(Debug)]
pub(in crate::repr) struct Prune<'a, D> {
    reader: store::intrusive::Reader<'a>,
    distance: D,
    buffer: Vec<*const u8>,
}

impl<'a, D> Prune<'a, D> {
    pub(in crate::repr) fn new(reader: store::intrusive::Reader<'a>, distance: D) -> Self {
        Self {
            reader,
            distance,
            buffer: Vec::new(),
        }
    }

    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

unsafe impl<D> Send for Prune<'_, D> where D: Send {}
unsafe impl<D> Sync for Prune<'_, D> where D: Sync {}

impl<D> repr::Prune for Prune<'_, D>
where
    D: super::RawDistance,
{
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<repr::PruneKey>>,
    ) -> ANNResult<usize> {
        let mut counter = repr::PruneKey::counter();
        self.buffer.clear();
        self.buffer.reserve(items.len());

        for (id, key) in items {
            if let Some(v) = self.reader.read(id.into_usize()) {
                self.buffer.push(v.as_ptr());

                *key = Some(counter);

                // Potential overflow issue - but it's exceedingly unlikely that
                // someone will provide a prune list exceeding `u16::MAX`.
                //
                // In addition, `diskann` limits this bound as well.
                counter = counter.increment()?;
            }
        }

        Ok(counter.index())
    }

    fn evaluate(&self, a: repr::PruneKey, b: repr::PruneKey) -> f32 {
        let len = self.reader.bytes().value();
        let a = unsafe { std::slice::from_raw_parts(self.buffer[a.index()], len) };
        let b = unsafe { std::slice::from_raw_parts(self.buffer[b.index()], len) };

        self.distance.eval(a, b).unwrap()
    }
}
