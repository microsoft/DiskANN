/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    bits::{Binary, MutBitSlice, Representation},
    distances::Hamming,
    AsFunctor, CompressInto,
};

/// A simple, training-free binary quantizer.
///
/// The canonical implementation of compression with a binary quantizer maps negative values
/// to -1 (encoded as a bit 0) and positive values to 1. Distances can then be approximated
/// using the Hamming distance between the compressed vectors.
///
/// As a convenience `diskann_quantization::bits::SquaredL2` and
/// `diskann_quantization::bits::InnerProduct` may be used, which correctly dispatch to the
/// proper post-op for similarity scores versus mathematical values.
///
/// # Example
/// ```rust
/// use diskann_quantization::{
///     AsFunctor, CompressInto,
///     distances::Hamming,
///     binary::BinaryQuantizer,
///     bits::{BoxedBitSlice, Binary},
/// };
///
/// use diskann_utils::{Reborrow, ReborrowMut};
/// use diskann_vector::{
///     PureDistanceFunction, DistanceFunction, MathematicalValue,
/// };
///
/// let x = vec![-1, 1, 1, -1, 1];
/// let y = vec![1, -1, 1, -1, -1];
///
/// // Create a quantizer
/// let quantizer = BinaryQuantizer;
///
/// // Create output vectors for compression.
/// let mut bx = BoxedBitSlice::<1, Binary>::new_boxed(x.len());
/// let mut by = BoxedBitSlice::<1, Binary>::new_boxed(y.len());
///
/// // Do the compression.
/// quantizer.compress_into(x.as_slice(), bx.reborrow_mut()).unwrap();
/// quantizer.compress_into(y.as_slice(), by.reborrow_mut()).unwrap();
///
/// // Because our inputs are limited to -1 and 1, the compression is perfect.
/// assert_eq!(bx.get(0).unwrap(), x[0]);
/// assert_eq!(bx.get(1).unwrap(), x[1]);
///
/// // But the compressed vectors only consume a single byte.
/// assert_eq!(bx.bytes(), 1);
///
/// // Lets compute some distances!
/// assert_eq!(
///     Hamming::evaluate(bx.reborrow(), by.reborrow()).unwrap(),
///     MathematicalValue::<u32>::new(3)
/// );
///
/// // We can also use the `AsFunctor` trait if we want more uniformity.
/// let f: Hamming = quantizer.as_functor();
/// assert_eq!(
///     f.evaluate_similarity(bx.reborrow(), by.reborrow()).unwrap(),
///     MathematicalValue::<u32>::new(3)
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BinaryQuantizer;

/////////////////
// Compression //
/////////////////

impl<T> CompressInto<&[T], MutBitSlice<'_, 1, Binary>> for BinaryQuantizer
where
    T: PartialOrd + Default,
{
    type Error = std::convert::Infallible;
    type Output = ();

    /// Compress the source vector into a binary representation.
    ///
    /// This works by mapping positive numbers (as defined by `v > T::default()`) to 1 and
    /// negative numbers (as defined by `v <= T::default()`) to -1.
    ///
    /// # Panics
    ///
    /// Panics if `from.len() != into.len()`.
    fn compress_into(
        &self,
        from: &[T],
        mut into: MutBitSlice<'_, 1, Binary>,
    ) -> Result<(), Self::Error> {
        // Check 1
        assert_eq!(from.len(), into.len());
        from.iter().enumerate().for_each(|(i, v)| {
            // Note: Both 1 and -1 are in the domain of `Binary`.
            let v: u8 = if v > &T::default() {
                Binary::encode_unchecked(1)
            } else {
                Binary::encode_unchecked(-1)
            };

            // SAFETY: From check 1, we know that `i < into.len()`.
            unsafe { into.set_unchecked(i, v) };
        });
        Ok(())
    }
}

///////////////
// AsFunctor //
///////////////

impl AsFunctor<Hamming> for BinaryQuantizer {
    /// Return a [`crate::distances::Hamming`] functor for performing distance computations
    /// on bit vectors.
    fn as_functor(&self) -> Hamming {
        Hamming
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{views::Matrix, ReborrowMut};
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    use super::*;
    use crate::bits::{Binary, BoxedBitSlice};

    fn test_compression_impl(len: usize, rng: &mut StdRng) {
        let mut domain = [-10, -1, 0, 1, 10];
        let mut test_pattern = Matrix::<i32>::new(0, domain.len(), len);

        // Fill the test patterns randomly.
        for col in 0..len {
            domain.shuffle(rng);
            for row in 0..test_pattern.nrows() {
                test_pattern[(row, col)] = domain[row];
            }
        }

        let quantizer = BinaryQuantizer;
        let mut binary = BoxedBitSlice::<1, Binary>::new_boxed(len);
        for row in test_pattern.row_iter() {
            quantizer.compress_into(row, binary.reborrow_mut()).unwrap();

            // Check the compression.
            for (i, r) in row.iter().enumerate() {
                if *r > 0 {
                    assert_eq!(binary.get(i).unwrap(), 1);
                } else {
                    assert_eq!(binary.get(i).unwrap(), -1);
                }
            }
        }
    }

    #[test]
    fn test_compression() {
        let mut rng = StdRng::seed_from_u64(0x9673d0890bbb7231);
        for len in 1..17 {
            test_compression_impl(len, &mut rng);
        }
    }
}
