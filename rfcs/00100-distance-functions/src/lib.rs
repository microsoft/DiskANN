/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//////////////////////////////////////////////
// What are we returning from computations? //
//////////////////////////////////////////////

/// A result of a distance computation that has been transformed so that for resulting
/// distance `dx` and `dy`, `dx < dy` implies that `dx` is "more similar" than `dy`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimilarityScore<T>(T);

/// The mathematical result of a computation without any transformation for similarity.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MathematicalValue<T>(T);

//////////////////////////
// PureDistanceFunction //
//////////////////////////

/// Trait for distance function objects that behave like pure functions.
pub trait PureDistanceFunction<Left, Right, To = f32> {
    /// Evaluate the function on the left and right-hand arguments and return the result.
    fn evaluate(x: Left, y: Right) -> To;
}

///////////
// Impls //
///////////

/// Return the squared L2 distance between the arguments.
#[derive(Debug, Clone, Copy)]
pub struct SquaredL2;

/// Implement for fixed sized arrays.
impl PureDistanceFunction<&[f32], &[f32], f32> for SquaredL2 {
    fn evaluate(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&ix, &iy)| {
            let d = ix - iy;
            acc + d * d
        })
    }
}

/// Implement for fixed sized arrays.
impl<const N: usize> PureDistanceFunction<&[f32; N], &[f32; N], f32> for SquaredL2 {
    fn evaluate(x: &[f32; N], y: &[f32; N]) -> f32 {
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&ix, &iy)| {
            let d = ix - iy;
            acc + d * d
        })
    }
}

/// Implement for `f64` arguments.
impl PureDistanceFunction<&[f64], &[f64], f64> for SquaredL2 {
    fn evaluate(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0, |acc, (&ix, &iy)| {
            let d = ix - iy;
            acc + d * d
        })
    }
}

/// Return the inner product of the arguments.
#[derive(Debug, Clone, Copy)]
pub struct InnerProduct;

/// Implement returning a Mathematical Value
impl PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>> for InnerProduct {
    fn evaluate(x: &[f32], y: &[f32]) -> MathematicalValue<f32> {
        assert_eq!(x.len(), y.len());
        let r: f32 =
            std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&ix, &iy)| acc + ix * iy);
        MathematicalValue(r)
    }
}

/// Implement returning a similarity score (applying a post-op to the mathematical result.
impl PureDistanceFunction<&[f32], &[f32], SimilarityScore<f32>> for InnerProduct {
    /// The implementation here works by invoking the method returning a mathematical value,
    /// unwrapping the inner value, applying a transformation, and returning the resulting
    /// similarity score.
    fn evaluate(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
        // Specify the return type to pick the correct overload.
        let r: MathematicalValue<f32> = Self::evaluate(x, y);
        SimilarityScore(-r.0)
    }
}

#[cfg(test)]
mod test_1 {
    use super::*;

    #[test]
    fn test_argument_overloading() {
        let x: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y: Vec<f32> = vec![2.0, 1.0, 0.0];

        // Squared L2 on dynamically sized vectors.
        let r: f32 = SquaredL2::evaluate(&*x, &*y);
        assert_eq!(r, 8.0);

        // Squared L2 on static arrays.
        let static_x = <[f32; 3]>::try_from(&*x).unwrap();
        let static_y = <[f32; 3]>::try_from(&*y).unwrap();

        // This call works and will specialize the kernel implementation on the fixed size
        // of the arguments.
        let r: f32 = SquaredL2::evaluate(&static_x, &static_y);
        assert_eq!(r, 8.0);

        // The same syntax works for `f64` as well.
        let x64: Vec<f64> = x.iter().map(|&i| i.into()).collect();
        let y64: Vec<f64> = y.iter().map(|&i| i.into()).collect();
        let r: f64 = SquaredL2::evaluate(&*x64, &*y64);
        assert_eq!(r, 8.0);
    }

    #[test]
    fn test_return_overloading() {
        let x: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y: Vec<f32> = vec![2.0, 1.0, 0.0];

        // Inner product returning the mathematical value.
        let r: MathematicalValue<f32> = InnerProduct::evaluate(&*x, &*y);
        assert_eq!(r, MathematicalValue(1.0));

        // Inner product returning a similarity score.
        let r: SimilarityScore<f32> = InnerProduct::evaluate(&*x, &*y);
        assert_eq!(r, SimilarityScore(-1.0));
    }
}

//////////////////////////////////
// PreprocessedDistanceFunction //
//////////////////////////////////

/// A common pattern for distance computations in similarity search is to have one argument
/// (usually the query) fixed across many distance computation (i.e., values for the other
/// argument).
///
/// For this stable argument, it can be beneficial to perform some amount of preprocessing
/// to accelerate similarity computation for the rapidly changing arguments.
///
/// In the case of Product Quantization (PQ), this preprocessing step can compute the
/// partial distances between the query chunks and the centers for each chunk. At which
/// point, distance computations for PQ encoded vectors become table lookups.
///
/// The `PreprocessedDistanceFunction` trait models an object that has performed this
/// pre-processing and is available to perform computations immediately.
///
/// Unlike the `PureDistanceFunction`, the evaluation method (`evaluate_similarity`) only
/// takes a single argument. This implies that the object implementing
/// `PreprocessDistanceFunction` must likely contain either a reference or a copy of the
/// stable, fixed argument.
pub trait PreprocessedDistanceFunction<Changing, To = f32> {
    /// Compute the similarity between the (potentially pre-processed) query inside `Self`
    /// and the rapidly `changing` right-hand argument.
    fn evaluate_similarity(&self, changing: Changing) -> To;
}

///////////
// Impls //
///////////

#[derive(Debug, Clone, Default)]
pub struct CosineStateful<'a> {
    query: &'a [f32],
    query_norm: f32,
}

impl<'a> CosineStateful<'a> {
    pub fn new(query: &'a [f32]) -> Self {
        Self {
            query,
            query_norm: query.iter().map(|&i| i * i).sum::<f32>().sqrt(),
        }
    }
}

impl<'a> PreprocessedDistanceFunction<&[f32]> for CosineStateful<'a> {
    fn evaluate_similarity(&self, changing: &[f32]) -> f32 {
        // Pre-check the self norm.
        if self.query_norm < (f32::EPSILON).sqrt() {
            return 0.0;
        }

        let (xy, ynorm) = std::iter::zip(self.query.iter(), changing.iter())
            .fold((0.0f32, 0.0f32), |(xy, ynorm), (&ix, &iy)| {
                (xy + ix * iy, ynorm + iy * iy)
            });

        if ynorm < f32::EPSILON {
            0.0
        } else {
            xy / (ynorm.sqrt() * self.query_norm)
        }
    }
}

/// A stateless version of the cosine similarity computation.
#[derive(Debug, Clone, Copy)]
pub struct CosineStateless;

impl PureDistanceFunction<&[f32], &[f32]> for CosineStateless {
    fn evaluate(x: &[f32], y: &[f32]) -> f32 {
        let init = (0.0f32, 0.0f32, 0.0f32);
        let (xy, xnorm, ynorm) = std::iter::zip(x.iter(), y.iter())
            .fold(init, |(xy, xnorm, ynorm), (&ix, &iy)| {
                (xy + ix * iy, xnorm + ix * ix, ynorm + iy * iy)
            });

        if ynorm < f32::EPSILON || xnorm < f32::EPSILON {
            0.0
        } else {
            xy / (ynorm.sqrt() * xnorm.sqrt())
        }
    }
}

#[cfg(test)]
mod test_3 {
    use super::*;

    #[test]
    fn test_stateful() {
        let x: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let y: Vec<f32> = vec![2.0, 0.0, 0.0, 0.0];
        let z: Vec<f32> = vec![-0.25, -0.25, -0.25, -0.25];

        // Compute the stateless version.
        let xy_stateless = CosineStateless::evaluate(&*x, &*y);
        let xz_stateless = CosineStateless::evaluate(&*x, &*z);

        // While this process may be slower for a single computation, the faster distance
        // computations it enables can be amortized over many computations.
        let stateful = CosineStateful::new(&x);
        let xy_stateful = stateful.evaluate_similarity(&*y);
        let xz_stateful = stateful.evaluate_similarity(&*z);

        assert_eq!(xy_stateless, xy_stateful);
        assert_eq!(xz_stateless, xz_stateful);
    }
}

///////////////////////////////////
// Creation of Function Pointers //
///////////////////////////////////

/// Obtain the `evaluate` method for the distance function as a function pointer.
pub fn as_function_pointer<T, Left, Right, Return>() -> fn(&[Left], &[Right]) -> Return
where
    T: for<'a, 'b> PureDistanceFunction<&'a [Left], &'b [Right], Return>,
{
    // Return a function pointer to a stateless closure.
    |x: &[Left], y: &[Right]| -> Return { T::evaluate(x, y) }
}

pub fn as_function_pointer_const<T, const N: usize, Left, Right, Return>(
) -> fn(&[Left], &[Right]) -> Return
where
    T: for<'a, 'b> PureDistanceFunction<&'a [Left; N], &'b [Right; N], Return>,
{
    |x: &[Left], y: &[Right]| -> Return {
        // Assert lengths are correct.
        assert_eq!(x.len(), N);
        assert_eq!(y.len(), N);

        // SAFETY: We have checked that both arguments have the correct length.
        //
        // The alignment requirements of arrays are the alignment requirements of
        // `Left` and `Right` respectively, which is provided by the corresponding slices.
        T::evaluate(unsafe { &*(x.as_ptr() as *const [Left; N]) }, unsafe {
            &*(y.as_ptr() as *const [Right; N])
        })
    }
}

#[cfg(test)]
mod test_2 {
    use super::*;

    #[test]
    fn test_function_pointer() {
        let x: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y: Vec<f32> = vec![2.0, 1.0, 0.0];

        let fptr: fn(&[f32], &[f32]) -> f32 = as_function_pointer::<SquaredL2, _, _, _>();
        assert_eq!(fptr(&x, &y), SquaredL2::evaluate(&*x, &*y));

        let fptr_const: fn(&[f32], &[f32]) -> f32 =
            as_function_pointer_const::<SquaredL2, 3, _, _, _>();
        assert_eq!(fptr_const(&x, &y), SquaredL2::evaluate(&*x, &*y));
    }
}

///////////////////////////////////////////////////////
// Automatic Implementation of PureDistanceFunctions //
///////////////////////////////////////////////////////

/// A marker trait indicating that a type implementing `PureDistanceFunction` wishes to
/// participate in automatic promotion to a `PreprocessedDistanceFunction`.
pub trait AutoPromoteMarker {}

/// Type implementing `PureDistanceFunctions` are generally expected to be zero sized types.
///
/// The automatic promotion strategy needs to maintain a reference to the stable argument.
/// The `PromotedPure` struct defined here is the promoted version of the pure function
/// that holds onto this reference.
///
/// We require `Stable: Copy` since it can be passed multiple times to `F::evaluate`.
///
/// Note that slices like `&[f32]` **are** `Copy`.
pub struct PromotedPure<F: AutoPromoteMarker, Stable: Copy> {
    stable: Stable,
    marker: std::marker::PhantomData<F>,
}

impl<F: AutoPromoteMarker, Stable: Copy> PromotedPure<F, Stable> {
    pub fn new(_marker: F, stable: Stable) -> Self {
        Self {
            stable,
            marker: std::marker::PhantomData,
        }
    }
}

impl<F, Stable, Changing, To> PreprocessedDistanceFunction<Changing, To> for PromotedPure<F, Stable>
where
    Stable: Copy,
    F: PureDistanceFunction<Stable, Changing, To> + AutoPromoteMarker,
{
    /// Implement `evaluate_similarity` by invoking `F::evaluate` with the cached version
    /// of `stable` and the provided `changing` argument.
    fn evaluate_similarity(&self, changing: Changing) -> To {
        F::evaluate(self.stable, changing)
    }
}

/// For exposition purposes, implement `AutoPromoteMarker` for `InnerProduct`.
/// See the test modules below for examples.
impl AutoPromoteMarker for InnerProduct {}

#[cfg(test)]
mod test_4 {
    use super::*;

    #[test]
    fn test_autopromote() {
        let x: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y: Vec<f32> = vec![2.0, 1.0, 0.0];

        let r: SimilarityScore<f32> = InnerProduct::evaluate(&*x, &*y);
        assert_eq!(r, SimilarityScore(-1.0));

        let ip = InnerProduct;
        let promoted = PromotedPure::new(ip, &*x);
        let r: SimilarityScore<f32> = promoted.evaluate_similarity(&*y);
        assert_eq!(r, SimilarityScore(-1.0));
    }
}
