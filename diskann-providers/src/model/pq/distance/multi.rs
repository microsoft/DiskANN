/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ops::Deref;

use diskann::ANNResult;
use diskann_utils::Reborrow;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use super::{
    QueryComputer,
    dynamic::{DistanceComputerConstructionError, VTable},
};
use crate::model::FixedChunkPQTable;

pub trait PQVersion: Eq + Copy {}
impl<T> PQVersion for T where T: Eq + Copy {}

/// A PQ vector with an associated version.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionedPQVector<I: PQVersion> {
    data: Vec<u8>,
    version: I,
}

impl<I> VersionedPQVector<I>
where
    I: PQVersion,
{
    /// Construct a new `VersionedPQVector` taking ownership of the provided data and version.
    pub fn new(data: Vec<u8>, version: I) -> Self {
        Self { data, version }
    }

    /// Return a `VersionedPQVectorRef` over the data owned by this vector.
    pub fn as_ref(&self) -> VersionedPQVectorRef<'_, I> {
        VersionedPQVectorRef::new(&self.data, self.version)
    }

    /// Return the version associated with this vector.
    pub fn version(&self) -> &I {
        &self.version
    }

    /// Return the raw underlying data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Return the components of the vector. This is a low-level API.
    pub fn raw_mut(&mut self) -> (&mut Vec<u8>, &mut I) {
        (&mut self.data, &mut self.version)
    }
}

impl<'a, I> Reborrow<'a> for VersionedPQVector<I>
where
    I: PQVersion,
{
    type Target = VersionedPQVectorRef<'a, I>;
    fn reborrow(&'a self) -> Self::Target {
        self.as_ref()
    }
}

/// A reference version of `VersionedPQVector`.
#[derive(Debug, Clone, Copy)]
pub struct VersionedPQVectorRef<'a, I: PQVersion> {
    data: &'a [u8],
    version: I,
}

impl<'a, I: PQVersion> VersionedPQVectorRef<'a, I> {
    /// Construct a new `VersionedPQVectorRef` around the provided data.
    pub fn new(data: &'a [u8], version: I) -> Self {
        Self { data, version }
    }

    /// Return the version associated with this vector.
    pub fn version(&self) -> &I {
        &self.version
    }

    /// Return the raw underlying data.
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

/// A wrapper for `FixedChunkPQTable` that contains either one or two inner
/// `FixedChunkPQTables` with associated versions.
#[derive(Debug, Clone)]
pub enum MultiTable<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    /// Only one table is present with an associated version.
    One { table: T, version: I },
    /// Two tables are present, an incoming "new" table and an outgoing "old" table.
    /// The versions of these tables are recorded respectively in `new_version` and
    /// `old_version`.
    Two {
        new: T,
        old: T,
        new_version: I,
        old_version: I,
    },
}

#[derive(Debug, Error)]
#[error("provided versions must not be equal")]
pub struct EqualVersionsError;

impl<T, I> MultiTable<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    /// Construct a new `MultiTable` containing a single `FixedChunkPQTable`.
    pub fn one(table: T, version: I) -> Self {
        Self::One { table, version }
    }

    /// Construct a new `MultiTable` with two `FixedChunkPQTable`s.
    ///
    /// Returns an `Err` if the two provided versions are equal.
    pub fn two(new: T, old: T, new_version: I, old_version: I) -> Result<Self, EqualVersionsError> {
        if new_version == old_version {
            Err(EqualVersionsError)
        } else {
            Ok(Self::Two {
                new,
                old,
                new_version,
                old_version,
            })
        }
    }

    /// Return the versions associated with the tables in this schema.
    ///
    /// The returned tuple depends on whether this table has one or two registerd schemas.
    ///
    /// * If there is only one schema, return the `(version, None)` where `version` is the
    ///   version of the only schema.
    /// * If there are two schema, return `(new_version, Some(old_version))` where
    ///   `new_version` is the version of the most recently registered schema while
    ///   `old_version` is the old version.
    pub fn versions(&self) -> (&I, Option<&I>) {
        match &self {
            Self::One { version, .. } => (version, None),
            Self::Two {
                new_version,
                old_version,
                ..
            } => (new_version, Some(old_version)),
        }
    }
}

/// A distance computer implementing
///
/// * `DistanceFunction<&[f32], &VersionedPQVector, Option<f32>`
/// * `DistanceFunction<&VersionedPQVector, &VersionedPQVector, Option<f32>`
///
/// That can contain either one or two PQ schemas, disambiguating which schema to use based
/// on the version numbers contained in the `VersionedPQVectors`.
///
/// Since this struct stores at most two PQ tables, that means there is the possibility
/// that a PQ vector is provided that does not match either of the tables.
///
/// Returns `None` for distance computations when the version of the PQ vector does not
/// match with any version in the local table.
#[derive(Debug, Clone)]
pub struct MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    table: MultiTable<T, I>,
    vtable: VTable,
}

impl<T, I> MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    /// Construct a `MultiDistanceComputer` from the provided table implementing the
    /// requested metric.
    pub fn new(
        table: MultiTable<T, I>,
        metric: Metric,
    ) -> Result<Self, DistanceComputerConstructionError> {
        // Check if OPQ is used. If so, we cannot correctly perform distance computations.
        match &table {
            MultiTable::One { table, .. } => {
                if table.has_opq() {
                    return Err(DistanceComputerConstructionError::OPQNotSupported);
                }
            }
            MultiTable::Two { new, old, .. } => {
                if new.has_opq() || old.has_opq() {
                    return Err(DistanceComputerConstructionError::OPQNotSupported);
                }
            }
        };
        Ok(Self {
            table,
            vtable: VTable::new(metric),
        })
    }

    /// Return the versions associated with the tables in this schema.
    ///
    /// The returned tuple depends on whether this table has one or two registerd schemas.
    ///
    /// * If there is only one schema, return the `(version, None)` where `version` is the
    ///   version of the only schema.
    /// * If there are two schema, return `(new_version, Some(old_version))` where
    ///   `new_version` is the version of the most recently registered schema while
    ///   `old_version` is the old version.
    pub fn versions(&self) -> (&I, Option<&I>) {
        self.table.versions()
    }
}

impl<T, I> DistanceFunction<&[f32], &VersionedPQVector<I>, Option<f32>>
    for MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: &[f32], y: &VersionedPQVector<I>) -> Option<f32> {
        self.evaluate_similarity(x, y.reborrow())
    }
}

impl<T, I> DistanceFunction<&[f32], VersionedPQVectorRef<'_, I>, Option<f32>>
    for MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    fn evaluate_similarity(&self, x: &[f32], y: VersionedPQVectorRef<'_, I>) -> Option<f32> {
        match &self.table {
            MultiTable::One { table, version } => {
                if version != &y.version {
                    None
                } else {
                    Some((self.vtable.distance_fn)(table, x, y.data))
                }
            }
            MultiTable::Two {
                old,
                new,
                old_version,
                new_version,
            } => {
                if old_version == &y.version {
                    Some((self.vtable.distance_fn)(old, x, y.data))
                } else if new_version == &y.version {
                    Some((self.vtable.distance_fn)(new, x, y.data))
                } else {
                    None
                }
            }
        }
    }
}

impl<T, I> DistanceFunction<&VersionedPQVector<I>, &VersionedPQVector<I>, Option<f32>>
    for MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    #[inline(always)]
    fn evaluate_similarity(
        &self,
        x: &VersionedPQVector<I>,
        y: &VersionedPQVector<I>,
    ) -> Option<f32> {
        self.evaluate_similarity(x.reborrow(), y.reborrow())
    }
}

/// Compute the distance between two versioned quantized vectors.
///
/// If one schema is currently being used and at least one of the versions of the argument
/// vectors does not match, return `None`.
///
/// If two schemas are used and at least one of the versions of the argument vectors is not
/// recognized, then return `None`.
impl<T, I> DistanceFunction<VersionedPQVectorRef<'_, I>, VersionedPQVectorRef<'_, I>, Option<f32>>
    for MultiDistanceComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    fn evaluate_similarity(
        &self,
        x: VersionedPQVectorRef<'_, I>,
        y: VersionedPQVectorRef<'_, I>,
    ) -> Option<f32> {
        match &self.table {
            MultiTable::One { table, version } => {
                if (&x.version != version) || (&y.version != version) {
                    None
                } else {
                    Some((self.vtable.distance_fn_qq)(table, x.data, y.data))
                }
            }
            MultiTable::Two {
                new,
                old,
                new_version,
                old_version,
            } => {
                let x_new = &x.version == new_version;
                let x_old = &x.version == old_version;

                let y_new = &y.version == new_version;
                let y_old = &y.version == old_version;

                if x_old {
                    if y_old {
                        // Both Old
                        Some((self.vtable.distance_fn_qq)(old, x.data, y.data))
                    } else if y_new {
                        let x_full = old.inflate_vector(x.data);
                        // X Old, Y New
                        Some((self.vtable.distance_fn)(new, &x_full, y.data))
                    } else {
                        None
                    }
                } else if x_new {
                    if y_old {
                        let y_full = old.inflate_vector(y.data);
                        // X New, Y Old
                        Some((self.vtable.distance_fn)(new, &y_full, x.data))
                    } else if y_new {
                        // Both new
                        Some((self.vtable.distance_fn_qq)(new, x.data, y.data))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
}

////////////////////
// Query Computer //
////////////////////

/// A `PreprocessedDistanceFunction` containing either one or two PQ schemas, capable of
/// performing distance computations with either. Upon a version mismatch with a query,
/// `None` is returned.
#[derive(Debug)]
pub enum MultiQueryComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    One {
        computer: QueryComputer<T>,
        version: I,
    },
    Two {
        new: QueryComputer<T>,
        old: QueryComputer<T>,
        new_version: I,
        old_version: I,
    },
}

impl<T, I> MultiQueryComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    /// Construct a new `MultiQueryComputer` with the requested metric and query.
    pub fn new<U>(table: MultiTable<T, I>, metric: Metric, query: &[U]) -> ANNResult<Self>
    where
        U: Into<f32> + Copy,
    {
        let s = match table {
            MultiTable::One { table, version } => Self::One {
                computer: { QueryComputer::new(table, metric, query, None)? },
                version,
            },
            MultiTable::Two {
                new,
                old,
                new_version,
                old_version,
            } => Self::Two {
                new: { QueryComputer::new(new, metric, query, None)? },
                old: { QueryComputer::new(old, metric, query, None)? },
                new_version,
                old_version,
            },
        };
        Ok(s)
    }

    /// Return a tuple that is either:
    /// 1. (The only table version, None)
    /// 2. (New Table Version, Old Table Version)
    pub fn versions(&self) -> (&I, Option<&I>) {
        match &self {
            Self::One { version, .. } => (version, None),
            Self::Two {
                new_version,
                old_version,
                ..
            } => (new_version, Some(old_version)),
        }
    }
}

impl<T, I> PreprocessedDistanceFunction<&VersionedPQVector<I>, Option<f32>>
    for MultiQueryComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    #[inline(always)]
    fn evaluate_similarity(&self, x: &VersionedPQVector<I>) -> Option<f32> {
        self.evaluate_similarity(x.reborrow())
    }
}

impl<T, I> PreprocessedDistanceFunction<VersionedPQVectorRef<'_, I>, Option<f32>>
    for MultiQueryComputer<T, I>
where
    T: Deref<Target = FixedChunkPQTable>,
    I: PQVersion,
{
    fn evaluate_similarity(&self, x: VersionedPQVectorRef<'_, I>) -> Option<f32> {
        match &self {
            Self::One { computer, version } => {
                if version != &x.version {
                    None
                } else {
                    Some(computer.evaluate_similarity(x.data))
                }
            }
            Self::Two {
                new,
                old,
                new_version,
                old_version,
            } => {
                if old_version == &x.version {
                    Some(old.evaluate_similarity(x.data))
                } else if new_version == &x.version {
                    Some(new.evaluate_similarity(x.data))
                } else {
                    None
                }
            }
        }
    }
}

/// # Testing Strategies.
///
/// ## Distance Computations
///
/// At this point, we assume that the lower level distance functions are more-or-less
/// accurate. That is, a given `QueryComputer` or VTable based distance work correctly.
///
/// The testing functions at this level are more designed for testing that versioned vectors
/// get sent to the right location and that the error handling is correct.
#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use approx::assert_relative_eq;
    use diskann::utils::{IntoUsize, VectorRepr};
    use diskann_vector::{Half, PreprocessedDistanceFunction};
    use rand::{Rng, SeedableRng, distr::Distribution};
    use rstest::rstest;

    use super::{
        super::test_utils::{self, TestDistribution},
        *,
    };

    fn to_f32<T>(x: &[T]) -> Vec<f32>
    where
        T: Into<f32> + Copy,
    {
        x.iter().map(|i| (*i).into()).collect()
    }

    /////////////////////////
    // Versioned PQ Vector //
    /////////////////////////

    #[test]
    fn test_versioned_pq_vector() {
        let vec = vec![1, 2, 3];
        let ptr = vec.as_ptr();
        let pq = VersionedPQVector::<usize>::new(vec, 10);
        assert_eq!(*pq.version(), 10);
        assert_eq!(pq.data().len(), 3);

        let data_ptr = pq.data().as_ptr();
        let pq_ref = pq.as_ref();
        assert_eq!(pq_ref.version(), pq.version());
        assert_eq!(data_ptr, ptr);
        assert_eq!(
            pq_ref.data().as_ptr(),
            data_ptr,
            "expected VersionedPQVectorRef to have the same underlying data as the \
             original VersionedPQVector"
        );

        let pq_ref = pq.reborrow();
        assert_eq!(pq_ref.version(), pq.version());
        assert_eq!(data_ptr, ptr);
        assert_eq!(
            pq_ref.data().as_ptr(),
            data_ptr,
            "expected VersionedPQVectorRef to have the same underlying data as the \
             original VersionedPQVector"
        );
    }

    ////////////////
    // MultiTable //
    ////////////////

    #[test]
    fn test_table_error() {
        let config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let new = test_utils::seed_pivot_table(config);
        let old = test_utils::seed_pivot_table(config);

        let result = MultiTable::two(&new, &old, 0, 0);
        assert!(
            matches!(result, Err(EqualVersionsError)),
            "MultiTable should now allow construction of the Two variant with equal versions"
        );
    }

    ///////////////////////////////////
    // Distance Computer - One Table //
    ///////////////////////////////////

    /// Test that the table works correctl where there is one inner PQ table.
    fn test_distance_computer_multi_with_one<R>(
        computer: &MultiDistanceComputer<&'_ FixedChunkPQTable, usize>,
        table: &FixedChunkPQTable,
        config: &test_utils::TableConfig,
        reference: &<f32 as VectorRepr>::Distance,
        num_trials: usize,
        rng: &mut R,
    ) where
        R: Rng,
    {
        // Check that there is just one version.
        let (&version, should_be_none) = computer.versions();
        assert!(
            should_be_none.is_none(),
            "expected just one schema in test computer"
        );
        let invalid_version = version.wrapping_add(1);

        for _ in 0..num_trials {
            let code0 = test_utils::generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let expected0 = test_utils::generate_expected_vector(
                &code0,
                table.get_chunk_offsets(),
                config.start_value,
            );

            let code1 = test_utils::generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let expected1 = test_utils::generate_expected_vector(
                &code1,
                table.get_chunk_offsets(),
                config.start_value,
            );

            let expected = reference.evaluate_similarity(&expected0, &expected1);

            // Test full-precision/quant.
            let got = computer
                .evaluate_similarity(&*expected0, &VersionedPQVector::new(code1.clone(), version))
                .expect("evaluate_similarity should return Some");
            assert_eq!(got, expected);

            let got = computer
                .evaluate_similarity(&*expected1, &VersionedPQVector::new(code0.clone(), version))
                .expect("evaluate_similarity should return Some");
            assert_eq!(got, expected);

            // Test quant/quant.
            let got = computer
                .evaluate_similarity(
                    &VersionedPQVector::new(code0.clone(), version),
                    &VersionedPQVector::new(code1.clone(), version),
                )
                .expect("evaluate_similarity should return Some");
            assert_eq!(got, expected);

            // Check that version mismatches return `None`.
            let got = computer.evaluate_similarity(
                &*expected0,
                &VersionedPQVector::new(code0.clone(), invalid_version),
            );
            assert!(got.is_none(), "version mismatches should return `None`");

            let got = computer.evaluate_similarity(
                &VersionedPQVector::new(code0.clone(), invalid_version),
                &VersionedPQVector::new(code1.clone(), version),
            );
            assert!(got.is_none(), "version mismatches should return `None`");

            let got = computer.evaluate_similarity(
                &VersionedPQVector::new(code0.clone(), version),
                &VersionedPQVector::new(code1.clone(), invalid_version),
            );
            assert!(got.is_none(), "version mismatches should return `None`");
        }
    }

    #[rstest]
    fn test_multi_distance_computer_one(
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc8da1164a88cef0f);

        let config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let table = test_utils::seed_pivot_table(config);

        let version: usize = 0x625b215f82f38008;

        let multi_table = MultiTable::one(&table, version);
        let (n, o) = multi_table.versions();
        assert_eq!(*n, version);
        assert!(o.is_none());

        let computer = MultiDistanceComputer::new(multi_table, metric).unwrap();

        test_distance_computer_multi_with_one(
            &computer,
            &table,
            &config,
            &f32::distance(metric, None),
            100,
            &mut rng,
        );
    }

    ////////////////////////////////////
    // Distance Computer - Two Tables //
    ////////////////////////////////////

    /// Test that the table works correctly when there are two inner PQ tables.
    #[allow(clippy::too_many_arguments)]
    fn test_distance_computer_multi_with_two<R>(
        computer: &MultiDistanceComputer<&'_ FixedChunkPQTable, usize>,
        new: &FixedChunkPQTable,
        old: &FixedChunkPQTable,
        new_config: &test_utils::TableConfig,
        old_config: &test_utils::TableConfig,
        reference: &<f32 as VectorRepr>::Distance,
        num_trials: usize,
        rng: &mut R,
    ) where
        R: Rng,
    {
        // Check that there are indeed two versions registered.
        let (&new_version, old_version) = computer.versions();
        let &old_version = old_version.expect("expected two schemas in test computer");

        for _ in 0..num_trials {
            // Generate a code for the old schema
            let old_code =
                test_utils::generate_random_code(old_config.num_pivots, old_config.pq_chunks, rng);
            let old_expected = test_utils::generate_expected_vector(
                &old_code,
                old.get_chunk_offsets(),
                old_config.start_value,
            );

            // Generate a code for the new schema
            let new_code =
                test_utils::generate_random_code(new_config.num_pivots, new_config.pq_chunks, rng);
            let new_expected = test_utils::generate_expected_vector(
                &new_code,
                new.get_chunk_offsets(),
                new_config.start_value,
            );

            // Generate reference results.
            let oo = reference.evaluate_similarity(&old_expected, &old_expected);
            let nn = reference.evaluate_similarity(&new_expected, &new_expected);
            let on = reference.evaluate_similarity(&old_expected, &new_expected);

            // Quant + Quant
            {
                let got_oo_qq = computer.evaluate_similarity(
                    &VersionedPQVector::new(old_code.clone(), old_version),
                    &VersionedPQVector::new(old_code.clone(), old_version),
                );
                assert_eq!(got_oo_qq.unwrap(), oo);

                let got_on_qq = computer.evaluate_similarity(
                    &VersionedPQVector::new(old_code.clone(), old_version),
                    &VersionedPQVector::new(new_code.clone(), new_version),
                );
                assert_eq!(got_on_qq.unwrap(), on);

                let got_no_qq = computer.evaluate_similarity(
                    &VersionedPQVector::new(new_code.clone(), new_version),
                    &VersionedPQVector::new(old_code.clone(), old_version),
                );
                assert_eq!(got_no_qq.unwrap(), on);

                let got_nn_qq = computer.evaluate_similarity(
                    &VersionedPQVector::new(new_code.clone(), new_version),
                    &VersionedPQVector::new(new_code.clone(), new_version),
                );
                assert_eq!(got_nn_qq.unwrap(), nn);
            }

            // Full Precision + Quant
            {
                let got_oo_qq = computer.evaluate_similarity(
                    &*old_expected,
                    &VersionedPQVector::new(old_code.clone(), old_version),
                );
                assert_eq!(got_oo_qq.unwrap(), oo);

                let got_on_qq = computer.evaluate_similarity(
                    &*old_expected,
                    &VersionedPQVector::new(new_code.clone(), new_version),
                );
                assert_eq!(got_on_qq.unwrap(), on);

                let got_no_qq = computer.evaluate_similarity(
                    &*new_expected,
                    &VersionedPQVector::new(old_code.clone(), old_version),
                );
                assert_eq!(got_no_qq.unwrap(), on);

                let got_nn_qq = computer.evaluate_similarity(
                    &*new_expected,
                    &VersionedPQVector::new(new_code.clone(), new_version),
                );
                assert_eq!(got_nn_qq.unwrap(), nn);
            }

            // Ensure that version mismatches return `None` for all combinations.
            let mut bad_version = old_version.wrapping_add(1);
            if bad_version == new_version {
                bad_version = bad_version.wrapping_add(1);
            }

            // mismatch for first argument.
            let got = computer.evaluate_similarity(
                VersionedPQVectorRef::new(&old_code, bad_version),
                VersionedPQVectorRef::new(&new_code, new_version),
            );
            assert!(got.is_none());

            // mismatch for second argument.
            let got = computer.evaluate_similarity(
                &VersionedPQVector::new(new_code.clone(), new_version),
                &VersionedPQVector::new(old_code.clone(), bad_version),
            );
            assert!(got.is_none());

            // mismatch for full precision.
            let got = computer.evaluate_similarity(
                &*new_expected,
                &VersionedPQVector::new(old_code.clone(), bad_version),
            );
            assert!(got.is_none());
        }
    }

    #[rstest]
    fn test_multi_distance_computer_two(
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc8da1164a88cef0f);

        let old_config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let new_config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 5,
            num_pivots: 16,
            start_value: 1.0,
            use_opq: false,
        };

        let new = test_utils::seed_pivot_table(new_config);
        let old = test_utils::seed_pivot_table(old_config);

        let new_version: usize = 0x5a2b92a731766613;
        let old_version: usize = 0x2fab58c9c8b73841;

        let multi_table = MultiTable::two(&new, &old, new_version, old_version).unwrap();
        let (n, o) = multi_table.versions();
        assert_eq!(*n, new_version);
        assert_eq!(*o.unwrap(), old_version);

        let computer = MultiDistanceComputer::new(multi_table.clone(), metric).unwrap();
        test_distance_computer_multi_with_two(
            &computer,
            &new,
            &old,
            &new_config,
            &old_config,
            &f32::distance(metric, None),
            100,
            &mut rng,
        );
    }

    ///////////////////////////////////////////
    // Distance Computer Construction Errors //
    ///////////////////////////////////////////

    #[rstest]
    fn test_multi_distance_computer_opq_error(
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) {
        let config_with_opq = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: true,
        };

        let config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let expected_err = (DistanceComputerConstructionError::OPQNotSupported).to_string();
        let table_with_opq = test_utils::seed_pivot_table(config_with_opq);
        let table = test_utils::seed_pivot_table(config);

        let schema = MultiTable::one(&table_with_opq, 0);
        let result = MultiDistanceComputer::new(schema, metric);
        assert!(result.is_err(), "expected OPQ to not be supported");
        assert_eq!(result.unwrap_err().to_string(), expected_err);

        // Try all combinations of tables with OPQ.
        let schema = MultiTable::two(&table_with_opq, &table, 0, 1).unwrap();
        let result = MultiDistanceComputer::new(schema, metric);
        assert!(result.is_err(), "expected OPQ to not be supported");
        assert_eq!(result.unwrap_err().to_string(), expected_err);

        let schema = MultiTable::two(&table, &table_with_opq, 0, 1).unwrap();
        let result = MultiDistanceComputer::new(schema, metric);
        assert!(result.is_err(), "expected OPQ to not be supported");
        assert_eq!(result.unwrap_err().to_string(), expected_err);

        let schema = MultiTable::two(&table_with_opq, &table_with_opq, 0, 1).unwrap();
        let result = MultiDistanceComputer::new(schema, metric);
        assert!(result.is_err(), "expected OPQ to not be supported");
        assert_eq!(result.unwrap_err().to_string(), expected_err);
    }

    ////////////////////////////////
    // Query Computer - One Table //
    ////////////////////////////////

    #[allow(clippy::too_many_arguments)]
    fn check_query_computer<R: Rng>(
        computer: &MultiQueryComputer<&'_ FixedChunkPQTable, usize>,
        table: &FixedChunkPQTable,
        config: &test_utils::TableConfig,
        query: &[f32],
        version: usize,
        rng: &mut R,
        reference: &<f32 as VectorRepr>::Distance,
        errors: test_utils::RelativeAndAbsolute,
    ) {
        // Generate a code for the old table.
        let code = test_utils::generate_random_code(config.num_pivots, config.pq_chunks, rng);
        let expected_vector = test_utils::generate_expected_vector(
            &code,
            table.get_chunk_offsets(),
            config.start_value,
        );
        let got = computer
            .evaluate_similarity(&VersionedPQVector {
                data: code,
                version,
            })
            .unwrap();
        let expected = reference.evaluate_similarity(query, &expected_vector);
        assert_relative_eq!(
            got,
            expected,
            epsilon = errors.absolute,
            max_relative = errors.relative
        );
    }

    fn test_query_computer_multi_with_one<'a, T, R>(
        mut create: impl FnMut(usize, &[T]) -> MultiQueryComputer<&'a FixedChunkPQTable, usize>,
        table: &'a FixedChunkPQTable,
        config: &test_utils::TableConfig,
        reference: &<f32 as VectorRepr>::Distance,
        num_trials: usize,
        rng: &mut R,
        errors: test_utils::RelativeAndAbsolute,
    ) where
        T: Into<f32> + TestDistribution,
        R: Rng,
    {
        let standard = rand::distr::StandardUniform {};
        for _ in 0..num_trials {
            let input: Vec<T> = T::generate(config.dim, rng);
            let input_f32 = to_f32(&input);

            let version: u64 = standard.sample(rng);
            let version: usize = version.into_usize();
            let invalid_version = version.wrapping_add(1);

            let computer = create(version, &input);

            assert_eq!(
                computer.versions(),
                (&version, None),
                "expected the computer to only have one version"
            );

            for _ in 0..num_trials {
                check_query_computer(
                    &computer, table, config, &input_f32, version, rng, reference, errors,
                );
            }

            // Check the error path on mismatched versions.
            let code = test_utils::generate_random_code(config.num_pivots, config.pq_chunks, rng);
            let got =
                computer.evaluate_similarity(VersionedPQVectorRef::new(&code, invalid_version));
            assert!(got.is_none(), "Expected `None` for unmatched versions");
        }
    }

    #[rstest]
    fn test_query_computer_one<T>(
        #[values(PhantomData::<f32>, PhantomData::<Half>, PhantomData::<u8>, PhantomData::<i8>)]
        _datatype: PhantomData<T>,
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) where
        T: Into<f32> + TestDistribution,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x6b53bef1bc26571e);

        let config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let table = test_utils::seed_pivot_table(config);
        let num_trials = 20;

        let errors = test_utils::RelativeAndAbsolute {
            relative: 5.0e-5,
            absolute: 0.0,
        };

        let create = |version: usize, query: &[T]| {
            let schema = MultiTable::one(&table, version);
            MultiQueryComputer::new(schema, metric, query).unwrap()
        };
        test_query_computer_multi_with_one(
            create,
            &table,
            &config,
            &f32::distance(metric, None),
            num_trials,
            &mut rng,
            errors,
        );
    }

    /////////////////////////////////
    // Query Computer - Two Tables //
    /////////////////////////////////

    #[allow(clippy::too_many_arguments)]
    fn test_query_computer_multi_with_two<'a, T, R>(
        create: impl Fn(usize, usize, &[T]) -> MultiQueryComputer<&'a FixedChunkPQTable, usize>,
        new: &'a FixedChunkPQTable,
        old: &'a FixedChunkPQTable,
        new_config: &test_utils::TableConfig,
        old_config: &test_utils::TableConfig,
        reference: &<f32 as VectorRepr>::Distance,
        num_trials: usize,
        rng: &mut R,
        errors: test_utils::RelativeAndAbsolute,
    ) where
        T: Into<f32> + TestDistribution,
        R: Rng,
    {
        let standard = rand::distr::StandardUniform {};
        for _ in 0..num_trials {
            let input: Vec<T> = T::generate(old_config.dim, rng);
            let input_f32: Vec<f32> = to_f32(&input);

            // Create a computer with two random versions.
            let old_version: u64 = standard.sample(rng);
            let mut new_version: u64 = standard.sample(rng);
            while new_version == old_version {
                new_version = standard.sample(rng);
            }

            let mut invalid_version: u64 = standard.sample(rng);
            while invalid_version == old_version || invalid_version == new_version {
                invalid_version = standard.sample(rng);
            }

            let old_version = old_version.into_usize();
            let new_version = new_version.into_usize();
            let invalid_version = invalid_version.into_usize();

            let computer = create(new_version, old_version, &input);

            assert_eq!(
                computer.versions(),
                (&new_version, Some(&old_version)),
                "versions were not propagated successfully",
            );

            for _ in 0..num_trials {
                check_query_computer(
                    &computer,
                    old,
                    old_config,
                    &input_f32,
                    old_version,
                    rng,
                    reference,
                    errors,
                );

                check_query_computer(
                    &computer,
                    new,
                    new_config,
                    &input_f32,
                    new_version,
                    rng,
                    reference,
                    errors,
                );

                let code = test_utils::generate_random_code(
                    old_config.num_pivots,
                    old_config.pq_chunks,
                    rng,
                );
                let got = computer.evaluate_similarity(&VersionedPQVector {
                    data: code,
                    version: invalid_version,
                });
                assert!(
                    got.is_none(),
                    "expected a distance computation with an invalid version to return None"
                );
            }
        }
    }

    #[rstest]
    fn test_query_computer_two<T>(
        #[values(PhantomData::<f32>, PhantomData::<Half>, PhantomData::<u8>, PhantomData::<i8>)]
        _datatype: PhantomData<T>,
        #[values(Metric::L2, Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) where
        T: Into<f32> + TestDistribution,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xc8da1164a88cef0f);

        let old_config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 4,
            num_pivots: 20,
            start_value: 10.0,
            use_opq: false,
        };

        let new_config = test_utils::TableConfig {
            dim: 17,
            pq_chunks: 5,
            num_pivots: 16,
            start_value: 1.0,
            use_opq: false,
        };

        let old = test_utils::seed_pivot_table(old_config);
        let new = test_utils::seed_pivot_table(new_config);
        let num_trials = 20;

        let create = |new_version: usize, old_version: usize, query: &[T]| {
            let schema = MultiTable::two(&new, &old, new_version, old_version).unwrap();
            MultiQueryComputer::new(schema, metric, query).unwrap()
        };

        let errors = test_utils::RelativeAndAbsolute {
            relative: 5.0e-5,
            absolute: 0.0,
        };

        test_query_computer_multi_with_two(
            create,
            &new,
            &old,
            &new_config,
            &old_config,
            &f32::distance(metric, None),
            num_trials,
            &mut rng,
            errors,
        );
    }
}
