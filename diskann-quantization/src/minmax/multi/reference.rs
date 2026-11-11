/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::kernel::{MinMaxMaxSimKernel, validate};
use crate::{
    bits::{Representation, Unsigned},
    minmax::{Coefficients, DataRef, MinMaxMeta},
    multi_vector::{MatRef, MaxSimError, Overflow},
};
use diskann_utils::views::rowmajor::{self, Matrix};

/// Scalar correctness baseline, independent of the prepared matrix kernels.
#[derive(Debug)]
pub(super) struct ReferenceKernel<const A_BITS: usize, const B_BITS: usize> {
    query: rowmajor::Owned<u8>,
    dim: usize,
}

impl<const A_BITS: usize, const B_BITS: usize> ReferenceKernel<A_BITS, B_BITS>
where
    Unsigned: Representation<A_BITS>,
{
    pub(super) fn new(query: MatRef<'_, MinMaxMeta<A_BITS>>) -> Result<Self, Overflow> {
        Ok(Self {
            query: query.as_canonical_view()?.to_rowmajor_owned(),
            dim: query.repr().intrinsic_dim(),
        })
    }
}

impl<const A_BITS: usize, const B_BITS: usize> MinMaxMaxSimKernel<B_BITS>
    for ReferenceKernel<A_BITS, B_BITS>
where
    Unsigned: Representation<A_BITS> + Representation<B_BITS>,
{
    fn nrows(&self) -> usize {
        self.query.nrows()
    }

    #[expect(
        clippy::expect_used,
        reason = "canonical query rows and dimensions validated before iteration"
    )]
    fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        if !validate(self.nrows(), self.dim, document, scores)? {
            return Ok(());
        }
        for (bytes, best) in self.query.rows().zip(scores) {
            let query = DataRef::<A_BITS>::from_canonical_front(bytes, self.dim)
                .expect("canonical query row");
            *best = f32::MAX;
            let q = Coefficients::from(query.meta());
            for doc in document.rows() {
                let dot: u64 = (0..self.dim)
                    .map(|d| {
                        query.vector().get(d).expect("query code in bounds") as u64
                            * doc.vector().get(d).expect("document code in bounds") as u64
                    })
                    .sum();
                let ip = q.compensate_inner_product(doc.meta().into(), dot as f32, self.dim as f32);
                *best = best.min(-ip);
            }
        }
        Ok(())
    }
}
