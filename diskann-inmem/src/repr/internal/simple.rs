/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult, neighbor::Neighbor, utils::IntoUsize};
use diskann_vector::{UnalignedSlice, distance::Distance};

use crate::{
    repr, store,
};

use super::{Calf, RawQueryDistance};

// TODO: Temporary Definition. Unify with `Full`.
#[derive(Debug)]
pub(in crate::repr) struct Temporary<'a, T, U>
where
    T: 'static,
    U: 'static,
{
    query: Calf<'a, [T]>,
    distance: Distance<T, U>,
}

impl<'a, T, U> Temporary<'a, T, U> {
    pub(in crate::repr) fn new(query: Calf<'a, [T]>, distance: Distance<T, U>) -> Self {
        Self { query, distance }
    }
}

impl<T, U> RawQueryDistance for Temporary<'_, T, U>
where
    T: std::fmt::Debug + Send + Sync + 'static,
    U: std::fmt::Debug + Send + Sync + 'static,
{
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        Ok(self
            .distance
            .call_unaligned(UnalignedSlice::from(&*self.query), unsafe {
                UnalignedSlice::new(x.as_ptr().cast::<U>(), x.len() / std::mem::size_of::<U>())
            }))
    }
}

//////////////
// Reranker //
//////////////

#[derive(Debug)]
pub(in crate::repr) struct Reranker<'a, D> {
    reader: store::simple::Reader<'a>,
    distance: D,
}

impl<'a, D> Reranker<'a, D> {
    pub(in crate::repr) fn new(reader: store::simple::Reader<'a>, distance: D) -> Self {
        Self { reader, distance }
    }
}

impl<'a, D> repr::PostProcess for Reranker<'a, D>
where
    D: RawQueryDistance,
{
    fn post_process(&mut self, buffer: &mut Vec<Neighbor<u32>>) -> ANNResult<()> {
        let mut result: Result<(), D::Error> = Ok(());
        buffer.retain_mut(|neighbor| {
            if result.is_err() {
                return false;
            }

            let id = *neighbor.id();
            if let Some(data) = self.reader.read(id.into_usize()) {
                match self.distance.eval(data) {
                    Ok(distance) => *neighbor = Neighbor::new(id, distance),
                    Err(err) => result = Err(err),
                }
                true
            } else {
                false
            }
        });

        buffer.sort_unstable_by(diskann::neighbor::ord::fast_distance);
        result.map_err(ANNError::new)
    }
}
