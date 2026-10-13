/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::ErrorContext};
use diskann_quantization::{distances as quant_distances, product::tables};
use diskann_utils::{
    lazy_format,
    object_pool::{self, ObjectPool},
    views::rowmajor::{self, Matrix, MatrixMut},
};

use crate::{num::Bytes, repr::generic};

#[derive(Debug)]
pub struct Product {
    /// The PQ table representation used for compression and creation of query computers.
    transposed: tables::TransposedTable,
    /// The PQ table representation used for pruning.
    padded: tables::PaddedTable,
    metric: Metric,

    /// An object pool for distance lookup tables.
    ///
    /// These involve non-trivial allocations, so it can be beneficial to hold onto them.
    cache: ObjectPool<DistanceTable>,
}

impl Product {
    pub fn new(
        table: tables::BasicTable,
        metric: Metric,
        cache_size: Option<NonZeroUsize>,
    ) -> ANNResult<Self> {
        let transposed = tables::TransposedTable::from_parts(
            table.view_pivots(),
            table.view_offsets().to_owned(),
        )
        .map_err(ANNError::new)
        .context("this is a broken internal invariant - please report")?;

        let padded = tables::PaddedTable::from_basic(table.as_view());

        let cache = ObjectPool::with_capacity(Some(cache_size.map(|v| v.get()).unwrap_or(0)));

        Ok(Self {
            transposed,
            padded,
            metric,
            cache,
        })
    }
}

impl generic::Quantizer for Product {
    fn bytes(&self) -> Bytes {
        Bytes::new(self.transposed.nchunks())
    }
    fn dim(&self) -> usize {
        self.transposed.dim()
    }
    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
        use diskann_quantization::CompressInto;

        let vlen = data.len();
        let dim = self.dim();
        if vlen != dim {
            return Err(ANNError::message(lazy_format!(
                move,
                "vector dim {} does not match quantizer dim {}",
                vlen,
                dim
            )));
        }

        self.transposed
            .compress_into(data, quantized)
            .map_err(ANNError::new)?;
        Ok(())
    }

    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: generic::DistanceVisitor<'a>,
    {
        visitor.visit(Distance::new(&self.padded, self.metric))
    }

    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: generic::QueryDistanceVisitor<'a>,
    {
        use diskann_vector::{Norm, norm::FastL2Norm};

        let args = DistanceTableArgs {
            nchunks: self.transposed.nchunks(),
            ncenters: self.transposed.ncenters(),
            metric: self.metric,
        };

        let mut distance_table = self.cache.get_ref(args);

        match &mut *distance_table {
            DistanceTable::SquaredL2(table) => {
                self.transposed
                    .process_into::<quant_distances::SquaredL2, _>(query, table.as_view_mut());
            }
            DistanceTable::InnerProduct(table) => {
                self.transposed
                    .process_into::<quant_distances::InnerProduct, _>(query, table.as_view_mut());
            }
            DistanceTable::Cosine { table, query_norm } => {
                self.transposed
                    .process_into::<quant_distances::Cosine, _>(query, table.as_view_mut());
                *query_norm = (FastL2Norm).evaluate(query);
            }
        }

        Ok(visitor.visit(QueryComputer(distance_table)))
    }
}

/// Distance metric to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl Metric {
    fn as_padded_metric(&self) -> tables::padded::Metric {
        match self {
            Self::SquaredL2 => tables::padded::Metric::SquaredL2,
            Self::InnerProduct => tables::padded::Metric::InnerProduct,
            Self::Cosine => tables::padded::Metric::Cosine,
        }
    }
}

impl From<diskann_vector::distance::Metric> for Metric {
    fn from(m: diskann_vector::distance::Metric) -> Self {
        use diskann_vector::distance::Metric as VMetric;

        match m {
            VMetric::L2 => Self::SquaredL2,
            VMetric::InnerProduct => Self::InnerProduct,
            VMetric::Cosine | VMetric::CosineNormalized => Self::Cosine,
        }
    }
}

impl From<generic::Metric> for Metric {
    fn from(m: generic::Metric) -> Self {
        match m {
            generic::Metric::SquaredL2 => Self::SquaredL2,
            generic::Metric::InnerProduct => Self::InnerProduct,
            generic::Metric::Cosine => Self::Cosine,
        }
    }
}

//----------------//
// Query Distance //
//----------------//

/// A populated distance table for query distances.
#[derive(Debug)]
enum DistanceTable {
    SquaredL2(rowmajor::Owned<f32>),
    InnerProduct(rowmajor::Owned<f32>),
    Cosine {
        table: rowmajor::Owned<tables::lookup::DotAndNorm>,
        query_norm: f32,
    },
}

#[derive(Debug, Clone, Copy)]
struct DistanceTableArgs {
    nchunks: usize,
    ncenters: usize,
    metric: Metric,
}

impl object_pool::AsPooled<DistanceTableArgs> for DistanceTable {
    fn create(args: DistanceTableArgs) -> Self {
        let DistanceTableArgs {
            nchunks,
            ncenters,
            metric,
        } = args;

        match metric {
            Metric::SquaredL2 => {
                DistanceTable::SquaredL2(rowmajor::Owned::from_element(nchunks, ncenters, 0.0))
            }
            Metric::InnerProduct => {
                DistanceTable::InnerProduct(rowmajor::Owned::from_element(nchunks, ncenters, 0.0))
            }
            Metric::Cosine => DistanceTable::Cosine {
                table: rowmajor::Owned::from_element(
                    nchunks,
                    ncenters,
                    tables::lookup::DotAndNorm::default(),
                ),
                query_norm: 0.0,
            },
        }
    }

    fn modify(&mut self, args: DistanceTableArgs) {
        let DistanceTableArgs {
            nchunks,
            ncenters,
            metric,
        } = args;

        let sizes_agree =
            |nrows: usize, ncols: usize| -> bool { nrows == nchunks && ncols == ncenters };

        let good = match (&self, metric) {
            (Self::SquaredL2(table), Metric::SquaredL2) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            (Self::InnerProduct(table), Metric::InnerProduct) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            (Self::Cosine { table, .. }, Metric::Cosine) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            _ => false,
        };

        if !good {
            *self = Self::create(args);
        }
    }
}

#[derive(Debug)]
struct QueryComputer<'a>(object_pool::PooledRef<'a, DistanceTable>);

impl generic::QueryDistance for QueryComputer<'_> {
    fn eval(&self, x: &[u8]) -> ANNResult<f32> {
        let distance = match &*self.0 {
            DistanceTable::SquaredL2(table) | DistanceTable::InnerProduct(table) => {
                tables::lookup::lookup_single(tables::lookup::Sum, table.as_view(), x)
                    .map_err(ANNError::new)?
            }
            DistanceTable::Cosine { table, query_norm } => {
                let sum = tables::lookup::lookup_single(tables::lookup::Sum, table.as_view(), x)
                    .map_err(ANNError::new)?;

                sum.finish_cosine(*query_norm).into_inner()
            }
        };

        Ok(distance)
    }
}

//----------//
// Distance //
//----------//

#[derive(Debug)]
struct Distance<'a> {
    table: &'a tables::PaddedTable,
    vtable: tables::padded::VTable,
}

impl<'a> Distance<'a> {
    fn new(table: &'a tables::PaddedTable, metric: Metric) -> Self {
        Self {
            table,
            vtable: table.vtable(metric.as_padded_metric()),
        }
    }
}

impl generic::Distance for Distance<'_> {
    fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        self.vtable
            .self_distance(self.table, x, y)
            .map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    //-------------//
    // Object Pool //
    //-------------//

    #[test]
    fn test_distance_table_as_pooled() {
        use object_pool::AsPooled;

        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 4,
            metric: Metric::SquaredL2,
        };
        let mut table = DistanceTable::create(args);

        let ptr = if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 4);
            table.as_ptr()
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify should leave the allocation untouched if it matches.
        table.modify(args);

        if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 4);
            assert_eq!(table.as_ptr(), ptr);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify works when changing sizes.
        let args = DistanceTableArgs {
            nchunks: 9,
            ncenters: 5,
            metric: Metric::SquaredL2,
        };
        table.modify(args);
        if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 9);
            assert_eq!(table.ncols(), 5);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify changes table type.
        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 6,
            metric: Metric::Cosine,
        };
        table.modify(args);
        if let DistanceTable::Cosine { ref table, .. } = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 6);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 6,
            metric: Metric::InnerProduct,
        };
        table.modify(args);
        if let DistanceTable::InnerProduct(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 6);
        } else {
            panic!("Unexpected table: {:?}", table);
        };
    }
}
