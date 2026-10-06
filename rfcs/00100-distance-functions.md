# Distance Function Interfaces - Rust API

The goal is to unify and simplify use of distance functions for both full precision and compressed vectors.
This PR describes proposed changes to the top level Rust API for general users of distance functions.
See the accompanying Rust crate for a compiling example of the code included here with some more details.

## Goals of the API

The main goal of this API is to make the way we interact with distance functions and quantized distance functions uniform to enable freedom of implementation and to enable new quantization techniques to be used in a drop-in manner.

This API tries to achieve the following:

* Safety: It should be easy to use and hard to misuse.
    Further, it should clearly delineate what is the result of a raw computation and what is a "similarity score" (see below).
    Additionally, I believe we should not require data to be aligned unless it is provably aligned through the type system.
* Ergonomics: It should enable generic code to be written, leveraging type inference and traits to provide a (more or less) uniform syntax when invoking functions.
    That is, there should be no types in function names.
* Expandability: The traits defined here like `PureDistanceFunction` and `DistanceFunction` should not only be applicable to uncompressed data, but directly applicable to quantization schemas like PQ, table based PQ, and future methods.
* Compatibility: The current DiskANN infrastructure uses distance providers to convert a metric to a function pointer.
    This API should be compatible with the existing infrastructure to reduce churn.
* Customizable: Provide general implementations of functions while providing the ability to request customizations in the implementation.
    These customizations could include:
    - Requests for strict numeric reproducibility.
    - Setting the low-water mark where a scalar implementation is used.

    Right now, I believe this can be done through generic parameters of the distance type and therefore done at compile time.
* Performant: Calling functions exposed by this API should be as fast as manually invoking the actual implementation.

## Notes on Performance

Here are some thoughts related to performance that I had in mind.

* We should be able to both *statically* and *dynamically* dispatch to implementations.
  For static dispatch, this means the compiler has full visibility into the destination of a call, allowing for inlining and potential call-site optimization.
  For dynamic dispatch, decayed function pointers should have no further indirection to the implementation.
  For both of these, we're relying on inlining to optimize through the dispatching logic.
  Furthermore, `dyn` traits allow `DistanceFunction` to be used in dynamic contexts.

* In some cases, knowing the exact dimensionality of the data to be used can result in much better code generation (often the cases for small vectors).
  We should endeavor to allow static-arrays to be used as if they were slices.
  Use of static arrays also provides a compile-time way of verifying that array lengths are correct.

* Often (especially in the context of Product Quantization), it can be beneficial to perform some pre-processing on a query, then reuse that work to accelerate future distance computations.
  One simple example is during the computation of `Cosine Similarity`.
  Our current implementations recompute the norm of the query on every single computation.
  However, the query's norm does not change during its lifetime.
  Instead, we can compute its norm *once*, and then reuse that through all computations.

  In the context of similarity search, it's commonly the case that queries remain static for the duration of many distance computations, so this is a natural fit.

# API Proposal

## SimilarityScore and MathematicalValue

In DiskANN, we often rank the results of distance computations using the `<` operator where we expect lower values to imply "more similar".
For distances like InnerProduct and Cosine (where by default, higher values mean more similar), this requires application of a transformation function to the result.
Common distances and transformations are listed below:

* L2: `x -> x` (no transformation)
* InnerProduct: `x -> -x`
* Cosine/CosineNormalized: `x -> 1 - x`

We can introduce and expose the following types to disambiguate the return values:

```rust
/// A result of a distance computation that has been transformed so that for resulting
/// distance `dx` and `dy`, `dx < dy` implies that `dx` is "more similar" than `dy`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimilarityScore<T>(T);

/// The mathematical result of a computation without any transformation for similarity.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MathematicalValue<T>(T);
```

**The current convention in DiskANN is that distance functions return similarity scores**.
This will be kept for the near future with overloads returning `SimilarityScore` and `MathematicalValue` coexisting.

Introducing return-type overloading does open the possibility of Rust inferring the incorrect type.
It may be worth not implementing operations like `PartialOrd` for `MathematicalValue` to prevent those from being accidentally used in ordered lists within `DiskANN`.

## PureDistanceFunction

Pure distance functions are [pure](https://en.wikipedia.org/wiki/Pure_function) in the sense that they do not contain local state, and always return the same result for the same arguments.
This is generally what we expect out of full-precision distance functions.
The trait modeling pure functions is given below.
```rust
/// Trait for distance function objects that behave like pure functions.
pub trait PureDistanceFunction<Left, Right, To = f32> {
    /// Evaluate the function on the left and right-hand arguments and return the result.
    fn evaluate(x: Left, y: Right) -> To;
}
```
This trait is overloaded on both argument types allowing for use cases like the following (see the accompanying code for implementation details).
```rust
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
```

Furthermore, the trait can also be overloaded on return type, enabling disambiguation of `f32`, `SimilarityScore`, `MathematicalValue`.
```rust
let x: Vec<f32> = vec![0.0, 1.0, 2.0];
let y: Vec<f32> = vec![2.0, 1.0, 0.0];

// Inner product returning the mathematical value.
let r: MathematicalValue<f32> = InnerProduct::evaluate(&*x, &*y);
assert_eq!(r, MathematicalValue(1.0));

// Inner product returning a similarity score.
let r: SimilarityScore<f32> = InnerProduct::evaluate(&*x, &*y);
assert_eq!(r, SimilarityScore(-1.0));
```

> **Note**
>
> To be clear on the semantics of pure functions, for a given CPU mirco-architecture and build version of DiskANN, the results of these functions should be repeatable regardless of the alignment of the arguments.
> However, I believe it is important to allow minor variations in outputs when moving across CPU micro-architectures or version of DiskANN.

### Discussion

The `PureDistanceFunction` trait provides a uniform calling syntax for distance functions with varying argument and return types.
This simplifies the use of such functions and avoids needing a unique name for each combination of argument types.
Behind the scenes, the implementation of, for example, `SquaredL2::evaluate` can dispatch to an implementation suitable for a given architecture and provide non-SIMD fallbacks if needed without caller code needing to be aware of such details.

Because the destination of these functions are statically dispatched, they become candidates for inlining and other inter-procedural optimizations.

Finally, if we wish to enable a power-user with call-site hints, we could do something like:
```rust
trait LengthHint {}

/// A hint that the vectors provided will be long.
struct HintLongVectors;
impl LengthHint for HintLongVectors {}

/// A hint that the vectors provided will be short.
struct HintShortVectors;
impl LengthHint for HintShortVectors {}

struct SquaredL2<Hint: LengthHint = HintLongVectors>;

// Implement `PureDistanceFunction` with each specified hint.
```

## `PreprocessedDistanceFunction` (extensions, including Quantization)

A common use pattern for similarity computations in similarity search applications is where one argument (usually the query) remains fixed across many similarity computations.
We can take advantage of this pattern to perform some pre-processing on the query if that will accelerate future distance computations.
The terminology we use here is that the argument `stable` or left-hand argument is kept constant for varying `changing`, or right-hand arguments.

One such example of where preprocessing can be helpful is in the evaluation of [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
The mathematical definition of cosine similarity involves the computation of the norm of both arguments.
However, if one of the arguments is stable, it can be beneficial to compute and store its norm once.

With that motivation, the `PreprocessedDistanceFunction` trait is defined as the following.
```rust
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
```
The expected use is that **upon construction**, an object `f` implementing `PreprocessedDistanceFunction` will make any of a reference, copy, or pre-processed copy of the query and be immediately useful for calls to `evaluate_similarity`.

An example implementation for cosine similarity would look like this.
```rust
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
```
With a use case like
```rust
let x: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
let y: Vec<f32> = vec![2.0, 0.0, 0.0, 0.0];
let z: Vec<f32> = vec![-0.25, -0.25, -0.25, -0.25];

// Compute the stateless version.
let xy_stateless = CosineStateless::evaluate(&*x, &*y);
let xz_stateless = CosineStateless::evaluate(&*x, &*z);

// While this process may be slower for a single computation, the faster distance
// computations it enables can be amortized over many computations.
let stateful = CosineStateful::new(&*x);
let xy_stateful = stateful.evaluate_similarity(&*y);
let xz_stateful = stateful.evaluate_similarity(&*z);

assert_eq!(xy_stateless, xy_stateful);
assert_eq!(xz_stateless, xz_stateful);
```

### Application to Quantization

A follow-up change demonstrates the use of this API for product quantization distance functions.
While quantized computers have some additional requirements (like attachment and detachment for paged search), the heavy lifting for distance computations is done by the `DistanceFunction` trait.

In the case of product quantization, preprocessing consists of computing the partial distances between the query and each pivot in the table.
The process of a distance computation then simply looks-up and sums these partial distances in an operation that is faster than a direct computation.

### Discussion

**Can this trait handle more complex scenarios?**:
I believe it can.
The [appendix](#extension) demonstrates how this trait can be used to handle dynamic requantization where quantized vectors can belong to one of two schemas at run time.

**Where does the pre-processing happen?**:
The `PreprocessedDistanceFunction` makes no statement about **how** the implementor is constructed nor where the pre-processing happens.
Furthermore, it does not prescribe if or how the intermediate scratchspace can be reused for multiple queries.
This is to provide implementors the greatest freedom when coming up with generic designs that can be layered on top of `PreprocessedDistanceFunction`.

However, it is expected that once a type implementing `PreprocessedDistanceFunction` is constructed, it is ready to go.
That is, implementers should make an unpreprocessed `PreprocessedDistanceFunction` unrepresentable.

## Initial Functions to be Exposed by Vector

The collection of functions to be exposed initially by the vector crate are:

* `SquaredL2`: The squared L2 distance between two vectors.
* `FullL2`: The L2 distance between two vectors (taking the square root of `SquaredL2`).
* `InnerProduct`: The inner product between two vectors.
* `Cosine`: Cosine similarity.
* `CosineNormalized`: Cosine similarity assuming the arguments are normalized.

These will be defined for the following types (where both left and right hand types are identical):
* `f32`
* `f16/Half`
* `i8`
* `u8`

For each of the following return types:

* `MathematicalValue<f32>`
* `SimilarityScore<f32>`
* `f32` with the same semantics as `SimilarityScore<f32>`.

### Numerical Semantics

* `i8/u8`: Arithmetic will be performed as-if each argument is converted to `i32` before any arithmetic is performed.
    This means that integer overflow will only happen for exceedingly long vectors.

    **This changes the behavior for L2 i8**: Our current implementation of L2 for `i8` arguments uses saturating arithmetic, which can easily saturate and return inaccurate results for arguments with a high dynamic range.

* `f32/Half`: Distances will be computed accurately, but the kernel implementation may vary depending on the CPU architecture.
    This is because floating point arithmetic is [non-associative](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-conditional-numerical-reproducibility-cnr.html), and the best strategy for implementing a distance function is dependent on the underlying architecture.

    A consequence of this is that values returned by floating point distance functions should not be persisted in a database.

    If we **do** need guaranteed numerically reproducibile results, we can add another hint or directive as described in the [pure distance function discussion](#hints).
    However, much care will be needed when defining these implementations so that the result is performant across all architectures of interest.

# Appendix

## Creation of Function Pointers

The distance provider in DiskANN is a factory for providing function pointers to distance functions based on a metric enum.
Types implementing `PureDistanceFunction` can be turned into function pointers as shown below:
```rust
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
```
An example of its use is shown below.
```rust
let x: Vec<f32> = vec![0.0, 1.0, 2.0];
let y: Vec<f32> = vec![2.0, 1.0, 0.0];

let fptr: fn(&[f32], &[f32]) -> f32 = as_function_pointer::<SquaredL2, _, _, _>();
assert_eq!(fptr(&x, &y), SquaredL2::evaluate(&*x, &*y));

let fptr_const: fn(&[f32], &[f32]) -> f32 = as_function_pointer_const::<SquaredL2, 3, _, _, _>();
assert_eq!(fptr_const(&x, &y), SquaredL2::evaluate(&*x, &*y));
```

## Automatic Promotion of PureDistanceFunctions

We can provide an opt-in blanket implementation for implementing `DistanceFunction` for any type that implements `PureDistanceFunction` as follows:
```rust
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

impl<'a, F, Stable, Changing, To> PreprocessedDistanceFunction<Changing, To>
    for PromotedPure<F, Stable>
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
```
Its use in code would look like this:
```rust
impl AutoPromoteMarker for InnerProduct {}

let x: Vec<f32> = vec![0.0, 1.0, 2.0];
let y: Vec<f32> = vec![2.0, 1.0, 0.0];


let r: SimilarityScore<f32> = InnerProduct::evaluate(&*x, &*y);
assert_eq!(r, SimilarityScore(-1.0));

let ip = InnerProduct;
let promoted = PromotedPure::new(ip, &*x);
let r: SimilarityScore<f32> = promoted.evaluate_similarity(&*y);
assert_eq!(r, SimilarityScore(-1.0));
```

## Nesting Property of DistanceFunction: Requantization

A follow-up change demonstrates the use of this API for product quantization distance functions.
While quantized computers have some additional requirements (like attachment and detachment for paged search), the heavy lifting for distance computations is done by the `DistanceFunction` trait.

To demonstrate the expandability of this approach, I can show how it can be augmented to support inline requantization.
**Inline Requantization** occurs when the DiskANN host is in the process of updating the PQ table schema and encodings in place, meaning that there are two active schemas (PQ tables) and a given PQ vector can belong to either schema.
Assuming the underlying schema implements `PureDistanceFunction`, then a sketch of implementing multiple schemas is given below:

```rust
struct MultiSchemaPQVector<'a> {
    codes: &'a [u8],
    /// An identifier for the schema this vector belongs to.
    /// Assume there are at most two active schemas.
    schema: u8,
}

/// A simplified representation of multiple concurrent schemas.
struct MultiSchemas {
    current: FixedChunkPQTable,
    next: FixedChunkPQTable,
    current_id: u8,
}

/// A computer for the multi-schema struct.
struct MultiSchemaComputer<'a> {
    // A computer for the current schema.
    current: pq::GenericComputer<'a>,
    // A computer for the next schema.
    next: pq::GenericComputer<'a>,
    current_id: u8,
}

impl<'a> MultiSchemaComputer<'a> {
    fn new<T: VectorElement>(schemas: &MultiSchemas, metric: Metric, query: &[T]) -> ANNResult<Self> {
        // Construct and pre-process.
        Ok(Self {
            current: create_scoped_functor(schemas.current, metric, query)?,
            next: create_scoped_functor(schemas.current, metric, query)?,
            current_id,
        })
    }
}

impl<'a> PreprocessedDistanceFunction<MultiSchemaPQVector<'_>, f32> for MultiSchemaComputer<'a> {
    fn evaluate_similarity(&self, code: MultiSchemaPQVector<'_>) -> f32 {
        // Dispatch to the correct schema.
        if code.schema == self.current_id {
            self.current.evaluate_similarity(query, code.codes)
        } else {
            self.next.evaluate_similarity(query, code.codes)
        }
    }
}
```
