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

use crate::{num::Bytes, repr::quantized};

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
    /// Construct a new [`Product`] quantizer.
    ///
    /// Lookup tables will be used to accelerate distance computations. Argument `cache_size`
    /// controls the number of such tables that will be held for reuse to reduce allocations.
    /// This should generally be set to the number of worker threads for best performance.
    /// Setting to `None` will disable distance table reuse.
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

impl quantized::Quantizer for Product {
    fn bytes(&self) -> Bytes {
        Bytes::new(self.transposed.nchunks())
    }

    fn dim(&self) -> usize {
        self.transposed.dim()
    }

    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
        use diskann_quantization::CompressInto;

        let data_len = data.len();
        let dim = self.dim();
        if data_len != dim {
            return Err(ANNError::message(lazy_format!(
                move,
                "vector dim {} does not match quantizer dim {}",
                data_len,
                dim
            )));
        }

        let qlen = quantized.len();
        let bytes = self.bytes().value();
        if qlen != bytes {
            return Err(ANNError::message(lazy_format!(
                move,
                "quantized buffer length {} does not match quantizer bytes {}",
                qlen,
                bytes,
            )));
        }

        self.transposed
            .compress_into(data, quantized)
            .map_err(ANNError::new)?;
        Ok(())
    }

    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: quantized::DistanceVisitor<'a>,
    {
        visitor.visit(Distance::new(&self.padded, self.metric))
    }

    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
    {
        use diskann_vector::{Norm, norm::FastL2Norm};

        let query_len = query.len();
        let dim = self.dim();
        if query_len != dim {
            return Err(ANNError::message(lazy_format!(
                move,
                "query length {} does not match quantizer dim {}",
                query_len,
                dim
            )));
        }

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
    #[cfg(test)]
    const fn all() -> &'static [Self] {
        &[Self::SquaredL2, Self::InnerProduct, Self::Cosine]
    }

    #[cfg(test)]
    fn as_vector_metric(&self) -> diskann_vector::distance::Metric {
        use diskann_vector::distance::Metric as VMetric;

        match self {
            Self::SquaredL2 => VMetric::L2,
            Self::InnerProduct => VMetric::InnerProduct,
            Self::Cosine => VMetric::Cosine,
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

impl From<quantized::Metric> for Metric {
    fn from(m: quantized::Metric) -> Self {
        match m {
            quantized::Metric::SquaredL2 => Self::SquaredL2,
            quantized::Metric::InnerProduct => Self::InnerProduct,
            quantized::Metric::Cosine => Self::Cosine,
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

impl quantized::QueryDistance for QueryComputer<'_> {
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
        use tables::padded::Metric as PMetric;

        let metric = match metric {
            Metric::SquaredL2 => PMetric::SquaredL2,
            Metric::InnerProduct => PMetric::InnerProduct,
            Metric::Cosine => PMetric::Cosine,
        };

        Self {
            table,
            vtable: table.vtable(metric),
        }
    }
}

impl quantized::Distance for Distance<'_> {
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

    use diskann::graph::test::synthetic::Grid;
    use diskann_utils::assert_contains;

    fn train(
        data: rowmajor::Ref<'_, f32>,
        chunks: usize,
        centers: usize,
        metric: Metric,
    ) -> Product {
        use diskann_quantization::{
            Parallelism,
            cancel::DontCancel,
            product::train::{self, TrainQuantizer},
            random,
            views::ChunkOffsets,
        };

        let trainer = train::LightPQTrainingParameters::new(centers, 2);
        let basic = trainer
            .train(
                data,
                ChunkOffsets::partition(
                    NonZeroUsize::new(data.ncols()).unwrap(),
                    NonZeroUsize::new(chunks).unwrap(),
                )
                .unwrap()
                .as_view(),
                Parallelism::Sequential,
                &random::StdRngBuilder::new(0),
                &DontCancel,
            )
            .unwrap();

        Product::new(basic, metric, NonZeroUsize::new(1)).unwrap()
    }

    #[derive(Debug)]
    struct Boxit;

    impl<'a> quantized::DistanceVisitor<'a> for Boxit {
        type Output = Box<dyn quantized::Distance + 'a>;
        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: quantized::Distance + 'a,
        {
            Box::new(distance)
        }
    }

    impl<'a> quantized::QueryDistanceVisitor<'a> for Boxit {
        type Output = Box<dyn quantized::QueryDistance + 'a>;
        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: quantized::QueryDistance + 'a,
        {
            Box::new(distance)
        }
    }

    // Use the canonical grid layout, but center the data around the origin.
    //
    // This allows cosine distances to return reasonable results as the data is distributed
    // around the origin.
    //
    // To keep computation mostly tractable, we only use a 2d grid with 9 points. So the
    // coordinates are as follows:
    //
    // 0:  [-1, -1]
    // 1:  [-1,  0]
    // 2:  [-1, +1]
    //
    // 3:  [ 0, -1]
    // 4:  [ 0,  0]
    // 5:  [ 0, +1]
    //
    // 6:  [+1, -1]
    // 7:  [+1,  0]
    // 8:  [+1, +1]
    fn make_test(metric: Metric) -> (Product, rowmajor::Owned<f32>) {
        let grid = Grid::Two;
        let mut data = grid.data(3);
        let offset = 1.0;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        // Train with 2 chunks and 5 centers. This should be sufficient to exactly represent
        // all the points in the grid.
        //
        // To ensure the results are exact, we append the start points to the training data.
        let product = train(data.as_view(), 2, 5, metric);

        assert_eq!(quantized::Quantizer::bytes(&product), Bytes::new(2));
        assert_eq!(quantized::Quantizer::dim(&product), data.ncols());

        (product, data)
    }

    fn test_distance(
        product: &Product,
        query: &[f32],
        data: rowmajor::Ref<'_, f32>,
        quantized: rowmajor::Ref<'_, u8>,
        reference: &dyn Fn(&[f32], &[f32]) -> f32,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(data.nrows(), quantized.nrows());

        let mut buf = vec![0u8; quantized::Quantizer::bytes(product).value()];
        quantized::Quantizer::quantize(product, &mut buf, query).unwrap();
        let distance = quantized::Quantizer::distance(product, Boxit);

        for (i, (data, quantized)) in std::iter::zip(data.rows(), quantized.rows()).enumerate() {
            let expected = reference(query, data);
            let got = distance.eval(&buf, quantized).unwrap();
            assert_eq!(
                expected, got,
                "failed on row {} with data {:?} -- {}",
                i, data, ctx
            );
        }
    }

    fn test_query_distance(
        product: &Product,
        query: &[f32],
        data: rowmajor::Ref<'_, f32>,
        quantized: rowmajor::Ref<'_, u8>,
        reference: &dyn Fn(&[f32], &[f32]) -> f32,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(data.nrows(), quantized.nrows());

        let distance = quantized::Quantizer::query_distance(product, query, Boxit).unwrap();

        for (i, (data, quantized)) in std::iter::zip(data.rows(), quantized.rows()).enumerate() {
            let expected = reference(query, data);
            let got = distance.eval(quantized).unwrap();
            assert_eq!(
                expected, got,
                "failed on row {} with data {:?} -- {}",
                i, data, ctx
            );
        }
    }

    /// Happy path tests.
    #[test]
    fn test_product() {
        use diskann_vector::distance::DistanceProvider;

        for metric in Metric::all() {
            let (product, data) = make_test(*metric);

            let reference =
                <f32 as DistanceProvider<f32>>::distance_comparer(metric.as_vector_metric(), None);

            // Quantize the data.
            let mut quantized = rowmajor::Owned::from_element(
                data.nrows(),
                quantized::Quantizer::bytes(&product).value(),
                0u8,
            );

            for (q, d) in std::iter::zip(quantized.rows_mut(), data.rows()) {
                quantized::Quantizer::quantize(&product, q, d).unwrap();
            }

            let mut queries = rowmajor::Owned::from_element(3, data.ncols(), 0.0);
            queries.row_mut(0).copy_from_slice(&[-1.0, -1.0]);
            queries.row_mut(1).copy_from_slice(&[1.0, 1.0]);
            queries.row_mut(2).copy_from_slice(&[0.0, 0.0]);

            for query in queries.rows() {
                test_distance(
                    &product,
                    query,
                    data.as_view(),
                    quantized.as_view(),
                    &|x, y| reference.call(x, y),
                    &format_args!("metric = {:?}, query = {:?}", metric, query),
                );

                test_query_distance(
                    &product,
                    query,
                    data.as_view(),
                    quantized.as_view(),
                    &|x, y| reference.call(x, y),
                    &format_args!("metric = {:?}, query = {:?}", metric, query),
                );
            }
        }
    }

    //--------//
    // Errors //
    //--------//

    #[test]
    fn test_quantize_errors() {
        let (product, _) = make_test(Metric::SquaredL2);

        assert_eq!(
            quantized::Quantizer::dim(&product),
            2,
            "tests tests are written with this assumption"
        );

        // Error for invalid `data` dim.
        {
            let mut quantized = [0, 0];
            let err =
                quantized::Quantizer::quantize(&product, &mut quantized, &[0.0; 3]).unwrap_err();
            assert_contains!(
                err.to_string(),
                "vector dim 3 does not match quantizer dim 2"
            );

            let err =
                quantized::Quantizer::quantize(&product, &mut quantized, &[0.0; 1]).unwrap_err();
            assert_contains!(
                err.to_string(),
                "vector dim 1 does not match quantizer dim 2"
            );

            let err = quantized::Quantizer::quantize(&product, &mut quantized, &[f32::INFINITY; 2])
                .unwrap_err();
            assert_contains!(err.to_string(), "infinity");
        }

        // Errors for invalid `quantized`.
        {
            let err = quantized::Quantizer::quantize(&product, &mut [0], &[0.0; 2]).unwrap_err();
            assert_contains!(
                err.to_string(),
                "quantized buffer length 1 does not match quantizer bytes 2"
            );

            let err = quantized::Quantizer::quantize(&product, &mut [0; 3], &[0.0; 2]).unwrap_err();
            assert_contains!(
                err.to_string(),
                "quantized buffer length 3 does not match quantizer bytes 2"
            );
        }
    }

    #[test]
    fn test_distance_errors() {
        for metric in Metric::all() {
            let (product, _) = make_test(*metric);

            assert_eq!(
                quantized::Quantizer::bytes(&product),
                Bytes::new(2),
                "tests tests are written with this assumption"
            );

            let distance = quantized::Quantizer::distance(&product, Boxit);

            let _ = distance.eval(&[0; 2], &[0; 1]).unwrap_err();
            let _ = distance.eval(&[0; 2], &[0; 3]).unwrap_err();
            let _ = distance.eval(&[0; 1], &[0; 2]).unwrap_err();
            let _ = distance.eval(&[0; 3], &[0; 2]).unwrap_err();
        }
    }

    #[test]
    fn test_query_distance_errors() {
        for metric in Metric::all() {
            let (product, _) = make_test(*metric);

            assert_eq!(
                quantized::Quantizer::bytes(&product),
                Bytes::new(2),
                "tests tests are written with this assumption"
            );

            // Invalid query
            {
                let err =
                    quantized::Quantizer::query_distance(&product, &[0.0], Boxit).unwrap_err();
                assert_contains!(
                    err.to_string(),
                    "query length 1 does not match quantizer dim 2"
                );

                let err =
                    quantized::Quantizer::query_distance(&product, &[0.0; 3], Boxit).unwrap_err();
                assert_contains!(
                    err.to_string(),
                    "query length 3 does not match quantizer dim 2"
                );
            }

            // Invalid buffer.
            let distance =
                quantized::Quantizer::query_distance(&product, &[0.0; 2], Boxit).unwrap();

            let _ = distance.eval(&[0; 1]).unwrap_err();
            let _ = distance.eval(&[0; 3]).unwrap_err();
        }
    }

    //--------//
    // Metric //
    //--------//

    #[test]
    fn test_metric_conversion() {
        use diskann_vector::distance::Metric as VMetric;
        use quantized::Metric as QMetric;

        assert_eq!(Metric::from(VMetric::L2), Metric::SquaredL2);
        assert_eq!(Metric::from(VMetric::InnerProduct), Metric::InnerProduct);
        assert_eq!(Metric::from(VMetric::Cosine), Metric::Cosine);
        assert_eq!(Metric::from(VMetric::CosineNormalized), Metric::Cosine);

        assert_eq!(Metric::from(QMetric::SquaredL2), Metric::SquaredL2);
        assert_eq!(Metric::from(QMetric::InnerProduct), Metric::InnerProduct);
        assert_eq!(Metric::from(QMetric::Cosine), Metric::Cosine);
    }

    //----------------//
    // Distance Table //
    //----------------//

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
