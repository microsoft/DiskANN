/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{Reborrow, ReborrowMut};
use diskann_vector::{MathematicalValue, PureDistanceFunction};
use thiserror::Error;

use crate::{
    bits::{BitSlice, Dense, Representation, Unsigned},
    distances,
    distances::{InnerProduct, MV},
    meta::{self},
};

/// A per-vector precomputed coefficients to help compute inner products
/// and squared L2 distances for the MinMax quantized vectors.
///
/// The inner product between `X = ax * X' + bx` and `Y = ay * Y' + by` for d-dimensional
/// vectors X and Y is:
/// ```math
/// <X, Y> = <ax * X' + bx, ay * Y' + by>
///        = ax * ay * <X', Y'> + ax * <X', by> + ay * <Y', bx> + d * bx * by.
/// ```
/// Let us define a grouping of these terms to make it easier to understand:
/// ```math
///  Nx = ax * sum_i X'[i],     Ny = ay * sum_i Y'[i],
/// ```
/// We can then simplify the dot product calculation as follows:
/// ```math
/// <X, Y> = ax * ay * <X', Y'> + Nx * by + Ny * bx +  d * bx * by
///                    --------
///                       |
///               Integer Dot Product
/// ```
///
/// To compute the squared L2 distance,
/// ```math
/// |X - Y|^2 = |ax * X' + bx|^2 + |ay * Y' + by|^2 - 2 * <X, Y>
/// ```
/// we can re-use the computation for inner-product from above.
#[derive(Default, Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct MinMaxCompensation {
    pub dim: u32,          // - dimension
    pub b: f32,            // - bx
    pub n: f32,            // - Nx
    pub a: f32,            // - ax
    pub norm_squared: f32, // - |ax * X' + bx|^2
}

const META_BYTES: usize = std::mem::size_of::<MinMaxCompensation>(); // This will be 5 * 4 = 20 bytes.

/// Error type for parsing a slice of bytes as a `DataRef`
/// and returning corresponding dimension.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MetaParseError {
    #[error("Invalid size: {0}, must contain at least {META_BYTES} bytes")]
    NotCanonical(usize),
}

impl MinMaxCompensation {
    /// Reads the dimension from the first 4 bytes of a MinMax quantized vector's metadata.
    ///
    /// This function is used to extract the vector dimension from serialized MinMax quantized
    /// vector data without fully deserializing the entire vector structure.
    ///
    /// # Arguments
    /// * `bytes` - A byte slice containing the serialized MinMax vector data
    ///
    /// # Returns
    /// * `Ok(dimension)` - The dimension of the vector as a `usize`
    /// * `Err(MetaParseError::NotCanonical(size))` - If the byte slice is shorter than 20 bytes (META_BYTES)
    ///
    /// # Usage
    /// Use this when you need to determine the vector dimension from serialized data before
    /// creating a `DataRef` or allocating appropriately sized buffers for decompression.
    #[inline(always)]
    pub fn read_dimension(bytes: &[u8]) -> Result<usize, MetaParseError> {
        if bytes.len() < META_BYTES {
            return Err(MetaParseError::NotCanonical(bytes.len()));
        }

        // SAFETY: There are at least `META_BYTES` = 20 bytes in the array so this access is within bounds.
        let dim_bytes: [u8; 4] = bytes.get(..4).map_or_else(
            || Err(MetaParseError::NotCanonical(bytes.len())),
            |slice| {
                slice
                    .try_into()
                    .map_err(|_| MetaParseError::NotCanonical(bytes.len()))
            },
        )?;

        let dim = u32::from_le_bytes(dim_bytes) as usize;

        Ok(dim)
    }
}

/// An owning compressed data vector
///
/// See: [`meta::Vector`].
pub type Data<const NBITS: usize> = meta::Vector<NBITS, Unsigned, MinMaxCompensation, Dense>;

/// A borrowed `Data` vector
///
/// See: [`meta::Vector`].
pub type DataRef<'a, const NBITS: usize> =
    meta::VectorRef<'a, NBITS, Unsigned, MinMaxCompensation, Dense>;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum DecompressError {
    #[error("expected src and dst length to be identical, instead src is {0}, and dst is {1}")]
    LengthMismatch(usize, usize),
}
impl<const NBITS: usize> DataRef<'_, NBITS>
where
    Unsigned: Representation<NBITS>,
{
    /// Decompresses a MinMax quantized vector back into its original floating-point representation.
    ///
    /// This method reconstructs the original vector values using the stored quantization parameters
    /// and the MinMax dequantization formula: `x = x' * a + b` and stores the result in `dst`
    ///
    /// # Arguments
    ///
    /// * `dst` - A mutable slice of `f32` values where the decompressed data will be written.
    ///   Must have the same length as the compressed vector.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - On successful decompression
    /// * `Err(DecompressError::LengthMismatch(src_len, dst_len))` - If the destination slice
    ///   length doesn't match the compressed vector length
    pub fn decompress_into(&self, dst: &mut [f32]) -> Result<(), DecompressError> {
        if dst.len() != self.len() {
            return Err(DecompressError::LengthMismatch(self.len(), dst.len()));
        }
        let meta = self.meta();

        // SAFETY: We checked that the length of the underlying vector is the same as
        // as `dst` so we are guaranteed to be within bounds when accessing the vector.
        dst.iter_mut().enumerate().for_each(|(i, d)| unsafe {
            *d = self.vector().get_unchecked(i) as f32 * meta.a + meta.b
        });
        Ok(())
    }
}

/// A mutable borrowed `Data` vector
///
/// See: [`meta::Vector`].
pub type DataMutRef<'a, const NBITS: usize> =
    meta::VectorMut<'a, NBITS, Unsigned, MinMaxCompensation, Dense>;

////////////////////
// Full Precision //
////////////////////

/// The inner product between `X = ax * X' + bx` and `Y` for d-dimensional
/// vectors X and Y is:
/// ```math
/// <X, Y> = <ax * X' + bx, Y>
///        = ax * <X', Y> + bx * sum(Y).
///               --------
///                  |
///          Integer-Float Dot Product
/// ```
///
/// To compute the squared L2 distance,
/// ```math
/// |X - Y|^2 = |ax * X' + bx|^2 + |Y|^2 - 2 * <X', Y>
/// ```
///
/// A Full Precision Query
#[derive(Debug)]
pub struct FullQuery {
    /// The data after transform is applied to it.
    pub data: Box<[f32]>,
    pub meta: FullQueryMeta,
}

/// A meta struct storing the `sum` and `norm_squared` of a
/// full query after transformation is applied to it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullQueryMeta {
    /// The sum of `data`.
    pub sum: f32,
    /// The norm of the 'data'.
    pub norm_squared: f32,
}

impl FullQuery {
    /// Construct an empty `FullQuery` for `dim` dimensional data.
    pub fn empty(dim: usize) -> Self {
        Self {
            data: vec![0.0f32; dim].into(),
            meta: Default::default(),
        }
    }

    /// Output the length of `data`
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Output if `data` is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'short> Reborrow<'short> for FullQuery {
    type Target = &'short FullQuery;
    fn reborrow(&'short self) -> Self::Target {
        self
    }
}

impl<'short> ReborrowMut<'short> for FullQuery {
    type Target = &'short mut FullQuery;
    fn reborrow_mut(&'short mut self) -> Self::Target {
        self
    }
}

///////////////////////////
// Compensated Distances //
///////////////////////////

fn kernel<const NBITS: usize, F>(
    x: DataRef<'_, NBITS>,
    y: DataRef<'_, NBITS>,
    f: F,
) -> distances::MathematicalResult<f32>
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
    F: Fn(f32, &MinMaxCompensation, &MinMaxCompensation) -> f32,
{
    let raw_product = InnerProduct::evaluate(x.vector(), y.vector())?;
    let (xm, ym) = (x.meta(), y.meta());
    let term0 = xm.a * ym.a * raw_product.into_inner() as f32;
    let term1_x = xm.n * ym.b;
    let term1_y = ym.n * xm.b;
    let term2 = xm.b * ym.b * (x.len() as f32);

    let v = term0 + term1_x + term1_y + term2;
    Ok(MV::new(f(v, &xm, &ym)))
}

pub struct MinMaxIP;

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::MathematicalResult<f32>>
    for MinMaxIP
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate(
        x: DataRef<'_, NBITS>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        kernel(x, y, |v, _, _| v)
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::Result<f32>>
    for MinMaxIP
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate(x: DataRef<'_, NBITS>, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let v: distances::MathematicalResult<f32> = Self::evaluate(x, y);
        Ok(-v?.into_inner())
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::MathematicalResult<f32>>
    for MinMaxIP
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        &'a [f32],
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &FullQuery, y: DataRef<'_, NBITS>) -> distances::MathematicalResult<f32> {
        let raw_product: f32 = InnerProduct::evaluate(&x.data, y.vector())?.into_inner();
        Ok(MathematicalValue::new(
            raw_product * y.meta().a + x.meta.sum * y.meta().b,
        ))
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::Result<f32>> for MinMaxIP
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        &'a [f32],
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &FullQuery, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let v: distances::MathematicalResult<f32> = Self::evaluate(x, y);
        Ok(-v?.into_inner())
    }
}

pub struct MinMaxL2Squared;

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::MathematicalResult<f32>>
    for MinMaxL2Squared
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate(
        x: DataRef<'_, NBITS>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        kernel(x, y, |v, xm, ym| {
            -2.0 * v + xm.norm_squared + ym.norm_squared
        })
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::Result<f32>>
    for MinMaxL2Squared
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate(x: DataRef<'_, NBITS>, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let v: distances::MathematicalResult<f32> = Self::evaluate(x, y);
        Ok(v?.into_inner())
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::MathematicalResult<f32>>
    for MinMaxL2Squared
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        &'a [f32],
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &FullQuery, y: DataRef<'_, NBITS>) -> distances::MathematicalResult<f32> {
        let raw_product = InnerProduct::evaluate(&x.data, y.vector())?.into_inner();

        let ym = y.meta();
        let compensated_ip = raw_product * ym.a + x.meta.sum * ym.b;
        Ok(MV::new(
            x.meta.norm_squared + ym.norm_squared - 2.0 * compensated_ip,
        ))
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::Result<f32>> for MinMaxL2Squared
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        &'a [f32],
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &FullQuery, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let v: distances::MathematicalResult<f32> = Self::evaluate(x, y);
        Ok(v?.into_inner())
    }
}

///////////////////////
// Cosine Distances //
///////////////////////

pub struct MinMaxCosine;

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::Result<f32>>
    for MinMaxCosine
where
    Unsigned: Representation<NBITS>,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
        DataRef<'a, NBITS>,
        DataRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    // 1 - <X, Y> / (|X| * |Y|)
    fn evaluate(x: DataRef<'_, NBITS>, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let ip: MV<f32> = MinMaxIP::evaluate(x, y)?;
        let (xm, ym) = (x.meta(), y.meta());
        Ok(1.0 - ip.into_inner() / (xm.norm_squared.sqrt() * ym.norm_squared.sqrt()))
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::Result<f32>> for MinMaxCosine
where
    Unsigned: Representation<NBITS>,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
        &'a FullQuery,
        DataRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &'_ FullQuery, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let ip: MathematicalValue<f32> = MinMaxIP::evaluate(x, y)?;
        let (xm, ym) = (x.meta.norm_squared, y.meta());
        Ok(1.0 - ip.into_inner() / (xm.sqrt() * ym.norm_squared.sqrt()))
        // 1 - <X, Y> / (|X| * |Y|)
    }
}

pub struct MinMaxCosineNormalized;

impl<const NBITS: usize>
    PureDistanceFunction<DataRef<'_, NBITS>, DataRef<'_, NBITS>, distances::Result<f32>>
    for MinMaxCosineNormalized
where
    Unsigned: Representation<NBITS>,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
        DataRef<'a, NBITS>,
        DataRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: DataRef<'_, NBITS>, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let ip: MathematicalValue<f32> = MinMaxIP::evaluate(x, y)?;
        Ok(1.0 - ip.into_inner()) // 1 - <X, Y>
    }
}

impl<const NBITS: usize>
    PureDistanceFunction<&FullQuery, DataRef<'_, NBITS>, distances::Result<f32>>
    for MinMaxCosineNormalized
where
    Unsigned: Representation<NBITS>,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
        &'a FullQuery,
        DataRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate(x: &'_ FullQuery, y: DataRef<'_, NBITS>) -> distances::Result<f32> {
        let ip: MathematicalValue<f32> = MinMaxIP::evaluate(x, y)?;
        Ok(1.0 - ip.into_inner()) // 1 - <X, Y>
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod minmax_vector_tests {
    use diskann_utils::Reborrow;
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::*;
    use crate::scalar::bit_scale;

    fn test_minmax_compensated_vectors<const NBITS: usize, R>(dim: usize, rng: &mut R)
    where
        Unsigned: Representation<NBITS>,
        InnerProduct: for<'a, 'b> PureDistanceFunction<
            BitSlice<'a, NBITS, Unsigned>,
            BitSlice<'b, NBITS, Unsigned>,
            distances::MathematicalResult<u32>,
        >,
        InnerProduct: for<'a, 'b> PureDistanceFunction<
            &'a [f32],
            BitSlice<'b, NBITS, Unsigned>,
            distances::MathematicalResult<f32>,
        >,
        R: Rng,
    {
        assert!(dim <= bit_scale::<NBITS>() as usize);

        // Create two vectors with known compensation values
        let mut v1 = Data::<NBITS>::new_boxed(dim);
        let mut v2 = Data::<NBITS>::new_boxed(dim);

        let domain = Unsigned::domain_const::<NBITS>();
        let code_distribution = Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap();

        // Set bit values
        {
            let mut bitslice1 = v1.vector_mut();
            let mut bitslice2 = v2.vector_mut();

            for i in 0..dim {
                bitslice1.set(i, code_distribution.sample(rng)).unwrap();
                bitslice2.set(i, code_distribution.sample(rng)).unwrap();
            }
        }
        let a_rnd = Uniform::new_inclusive(0.0, 2.0).unwrap();
        let b_rnd = Uniform::new_inclusive(0.0, 2.0).unwrap();

        // Set compensation coefficients
        // v1: X = a1 * X' + b1
        // v2: Y = a2 * Y' + b2
        let a1 = a_rnd.sample(rng);
        let b1 = b_rnd.sample(rng);
        let a2 = a_rnd.sample(rng);
        let b2 = b_rnd.sample(rng);

        // Calculate sum of vector elements for n values
        let sum1: f32 = (0..dim).map(|i| v1.vector().get(i).unwrap() as f32).sum();
        let sum2: f32 = (0..dim).map(|i| v2.vector().get(i).unwrap() as f32).sum();

        // Create original full-precision vectors for reference calculations
        let mut original1 = Vec::with_capacity(dim);
        let mut original2 = Vec::with_capacity(dim);

        // Calculate the reconstructed original vectors and their norms
        for i in 0..dim {
            let val1 = a1 * v1.vector().get(i).unwrap() as f32 + b1;
            let val2 = a2 * v2.vector().get(i).unwrap() as f32 + b2;
            original1.push(val1);
            original2.push(val2);
        }

        // Calculate squared norms
        let norm1_squared: f32 = original1.iter().map(|x| x * x).sum();
        let norm2_squared: f32 = original2.iter().map(|x| x * x).sum();

        // Set compensation coefficients
        v1.set_meta(MinMaxCompensation {
            a: a1,
            b: b1,
            n: a1 * sum1,
            norm_squared: norm1_squared,
            dim: dim as u32,
        });

        v2.set_meta(MinMaxCompensation {
            a: a2,
            b: b2,
            n: a2 * sum2,
            norm_squared: norm2_squared,
            dim: dim as u32,
        });

        // Calculate raw integer dot product
        let expected_ip = (0..dim).map(|i| original1[i] * original2[i]).sum::<f32>();

        // Test inner product with f32
        let computed_ip_f32: distances::Result<f32> =
            MinMaxIP::evaluate(v1.reborrow(), v2.reborrow());
        let computed_ip_f32 = computed_ip_f32.unwrap();
        assert!(
            (expected_ip - (-computed_ip_f32)).abs() / expected_ip.abs() < 1e-3,
            "Inner product (f32) failed: expected {}, got {} on dim : {}",
            -expected_ip,
            computed_ip_f32,
            dim
        );

        // Expected L2 distance = |X|² + |Y|² - 2<X,Y>
        let expected_l2 = (0..dim)
            .map(|i| original1[i] - original2[i])
            .map(|x| x.powf(2.0))
            .sum::<f32>();

        // Test L2 distance with f32
        let computed_l2_f32: distances::Result<f32> =
            MinMaxL2Squared::evaluate(v1.reborrow(), v2.reborrow());
        let computed_l2_f32 = computed_l2_f32.unwrap();
        assert!(
            ((computed_l2_f32 - expected_l2).abs() / expected_l2) < 1e-3,
            "L2 distance (f32) failed: expected {}, got {} on dim : {}",
            expected_l2,
            computed_l2_f32,
            dim
        );

        let expected_cosine = 1.0 - expected_ip / (norm1_squared.sqrt() * norm2_squared.sqrt());

        let computed_cosine: distances::Result<f32> =
            MinMaxCosine::evaluate(v1.reborrow(), v2.reborrow());
        let computed_cosine = computed_cosine.unwrap();

        {
            let passed = (computed_cosine - expected_cosine).abs() < 1e-6
                || ((computed_cosine - expected_cosine).abs() / expected_cosine) < 1e-3;

            assert!(
                passed,
                "Cosine distance (f32) failed: expected {}, got {} on dim : {}",
                expected_cosine, computed_cosine, dim
            );
        }

        let cosine_normalized: distances::Result<f32> =
            MinMaxCosineNormalized::evaluate(v1.reborrow(), v2.reborrow());
        let cosine_normalized = cosine_normalized.unwrap();
        let expected_cos_normalized = 1.0 - expected_ip;
        assert!(
            ((expected_cos_normalized - cosine_normalized).abs() / expected_cos_normalized.abs())
                < 1e-6,
            "CosineNormalized distance (f32) failed: expected {}, got {} on dim : {}",
            expected_cos_normalized,
            cosine_normalized,
            dim
        );

        //Calculate inner product with full precision vector
        let mut fp_query = FullQuery::empty(dim);
        let fp_meta = FullQueryMeta {
            norm_squared: norm1_squared,
            sum: original1.iter().sum::<f32>(),
        };
        fp_query.data = original1.clone().into_boxed_slice();
        fp_query.meta = fp_meta;

        let fp_ip: distances::Result<f32> = MinMaxIP::evaluate(fp_query.reborrow(), v2.reborrow());
        let fp_ip = fp_ip.unwrap();
        assert!(
            (expected_ip - (-fp_ip)).abs() / expected_ip.abs() < 1e-3,
            "Inner product (f32) failed: expected {}, got {} on dim : {}",
            -expected_ip,
            fp_ip,
            dim
        );

        let fp_l2: distances::Result<f32> =
            MinMaxL2Squared::evaluate(fp_query.reborrow(), v2.reborrow());
        let fp_l2 = fp_l2.unwrap();
        assert!(
            ((fp_l2 - expected_l2).abs() / expected_l2) < 1e-3,
            "L2 distance (f32) failed: expected {}, got {} on dim : {}",
            expected_l2,
            computed_l2_f32,
            dim
        );

        let fp_cosine: distances::Result<f32> =
            MinMaxCosine::evaluate(fp_query.reborrow(), v2.reborrow());
        let fp_cosine = fp_cosine.unwrap();
        let diff = (fp_cosine - expected_cosine).abs();
        assert!(
            (diff / expected_cosine) < 1e-3 || diff <= 1e-6,
            "Cosine distance (f32) failed: expected {}, got {} on dim : {}",
            expected_cosine,
            fp_cosine,
            dim
        );

        let fp_cos_norm: distances::Result<f32> =
            MinMaxCosineNormalized::evaluate(fp_query.reborrow(), v2.reborrow());
        let fp_cos_norm = fp_cos_norm.unwrap();
        assert!(
            (((1.0 - expected_ip) - fp_cos_norm).abs() / (1.0 - expected_ip)) < 1e-3,
            "Cosine distance (f32) failed: expected {}, got {} on dim : {}",
            (1.0 - expected_ip),
            fp_cos_norm,
            dim
        );

        //Test `decompress_into` to make sure it outputs tje full-precision vector correctly.
        let meta = v1.meta();
        let v1_ref = DataRef::new(v1.vector(), &meta);
        let dim = v1_ref.len();
        let mut boxed = vec![0f32; dim + 1];

        let pre = v1_ref.decompress_into(&mut boxed);
        assert_eq!(
            pre.unwrap_err(),
            DecompressError::LengthMismatch(dim, dim + 1)
        );
        let pre = v1_ref.decompress_into(&mut boxed[..dim - 1]);
        assert_eq!(
            pre.unwrap_err(),
            DecompressError::LengthMismatch(dim, dim - 1)
        );
        let pre = v1_ref.decompress_into(&mut boxed[..dim]);
        assert!(pre.is_ok());

        boxed
            .iter()
            .zip(original1.iter())
            .for_each(|(x, y)| assert!((*x - *y).abs() <= 1e-6));

        // Verify `read_dimension` is correct.
        let mut bytes = vec![0u8; Data::canonical_bytes(dim)];
        let mut data = DataMutRef::from_canonical_front_mut(bytes.as_mut_slice(), dim).unwrap();
        data.set_meta(meta);

        let pre = MinMaxCompensation::read_dimension(&bytes);
        assert!(pre.is_ok());
        let read_dim = pre.unwrap();
        assert_eq!(read_dim, dim);

        let pre = MinMaxCompensation::read_dimension(&[0_u8; 2]);
        assert_eq!(pre.unwrap_err(), MetaParseError::NotCanonical(2));
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The max dim does not need to be as high for `CompensatedVectors` because they
            // defer their distance function implementation to `BitSlice`, which is more
            // heavily tested.
            const TRIALS: usize = 2;
        } else {
            const TRIALS: usize = 10;
        }
    }

    macro_rules! test_minmax_compensated {
        ($name:ident, $nbits:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                const MAX_DIM: usize = (bit_scale::<$nbits>() as usize);
                for dim in 1..=MAX_DIM {
                    #[cfg(miri)]
                    if dim != MAX_DIM {
                        continue;
                    }

                    for _ in 0..TRIALS {
                        test_minmax_compensated_vectors::<$nbits, _>(dim, &mut rng);
                    }
                }
            }
        };
    }
    test_minmax_compensated!(unsigned_minmax_compensated_test_u1, 1, 0xa33d5658097a1c35);
    test_minmax_compensated!(unsigned_minmax_compensated_test_u2, 2, 0xaedf3d2a223b7b77);
    test_minmax_compensated!(unsigned_minmax_compensated_test_u4, 4, 0xf60c0c8d1aadc126);
    test_minmax_compensated!(unsigned_minmax_compensated_test_u8, 8, 0x09fa14c42a9d7d98);
}
