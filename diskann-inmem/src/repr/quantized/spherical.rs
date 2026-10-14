/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    alloc::{Allocator, GlobalAllocator, ScopedAllocator},
    spherical::iface,
};

use crate::{num::Bytes, repr::quantized};

impl<T> quantized::Quantizer for T
where
    T: iface::Quantizer + 'static,
{
    fn bytes(&self) -> Bytes {
        Bytes::new(iface::Quantizer::bytes(self))
    }
    fn dim(&self) -> usize {
        iface::Quantizer::full_dim(self)
    }
    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
        iface::Quantizer::compress(
            self,
            data,
            iface::OpaqueMut::new(quantized),
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)
    }
    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: quantized::DistanceVisitor<'a>,
    {
        visitor.visit(iface::Quantizer::distance_computer_ref(self))
    }
    fn insert_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
    {
        let query_computer = iface::Quantizer::fused_query_computer(
            self,
            query,
            iface::QueryLayout::SameAsData,
            false,
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)?;

        Ok(visitor.visit(query_computer))
    }
    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
    {
        let query_computer = iface::Quantizer::fused_query_computer(
            self,
            query,
            iface::QueryLayout::FullPrecision,
            true,
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)?;

        Ok(visitor.visit(query_computer))
    }
}

impl<A> quantized::QueryDistance for iface::QueryComputer<A>
where
    A: Allocator + std::fmt::Debug + Send + Sync,
{
    fn eval(&self, x: &[u8]) -> ANNResult<f32> {
        use diskann_vector::PreprocessedDistanceFunction;
        self.evaluate_similarity(iface::Opaque::new(x))
            .map_err(ANNError::new)
    }
}

impl quantized::Distance for &dyn iface::DynDistanceComputer {
    fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        self.evaluate(iface::Opaque::new(x), iface::Opaque::new(y))
            .map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::test::synthetic::Grid;
    use diskann_quantization::spherical::SupportedMetric;
    use diskann_utils::{assert_contains, views::rowmajor::MatrixMut};
    use diskann_vector::distance::Distance as VDistance;
    use hashbrown::HashMap;

    use crate::{
        counters::Counters,
        num::{LogicalId, SlotId},
        repr::test::{Reference, test_expand_beam, test_prune},
    };

    fn train_quantizer(
        data: rowmajor::Ref<'_, f32>,
        metric: SupportedMetric,
        bits: Bits,
    ) -> Poly<dyn iface::Quantizer> {
        use diskann_quantization::{algorithms::transforms, spherical};
        use rand::{SeedableRng, rngs::StdRng};

        let q = spherical::SphericalQuantizer::train(
            data.as_view(),
            transforms::TransformKind::Null,
            metric,
            spherical::PreScale::None,
            &mut StdRng::seed_from_u64(0x019823745),
            GlobalAllocator,
        )
        .unwrap();

        match bits {
            Bits::One => q.as_quantizer::<1>().unwrap(),
            Bits::Two => q.as_quantizer::<2>().unwrap(),
            Bits::Four => q.as_quantizer::<4>().unwrap(),
        }
    }
}
