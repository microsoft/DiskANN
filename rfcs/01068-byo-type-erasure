# Bring Your Own Type Erasure

| | |
|---|---|
| **Authors** | Mark Hildebrand |
| **Contributors** | |
| **Created** | 2026-05-14 |
| **Updated** | 2026-05-14 |

## Summary

This RFC outlines a pattern for tackling composition of distance computers with only a single level of type erasure.
The goal is to streamline patterns like #1050 where trait object based distance computers are embedded in new-type wrappers to create yet another trait object.

## Motivation

### Background

Lower level APIs in our library use various flavors of type-erasure to enable polymorphism over metric, micro-architecture, and length specialization.
This takes one of several forms:

* Function pointer (MinMax)
* `diskann-wide` magic function pointer (full-precision distances)
* Trait objects (spherical quantization in `iface.rs`)
* Enum matching (PQ)

While this is the right decision to avoid an absolute code explosion, it leads to an unfortunate composability problem.
Take for example the quantization approach taken in #1050 (adding quantization to `diskann-garnet`).
Here, an inner distance computations (using one of the type-erasure approaches outlined above) need to be composed with a small unwrapping layer.
For `diskann-garnet` specifically, this unwrapping layer reifies the type of raw byte slices (translates from `&[u8]` to the type needed by the inner computer).
The combination of unwrapping + delegation is used to create another trait object, leading to an unavoidable situations where we have at least two levels of dynamic dispatch.
A small diagram is shown below:
```
Box<dyn QueryDistance>                      // Outer trait object
 |
 +-- Some small amount of work
     Inner: Box<dyn SomeOtherQueryDistance> // Inner trait object
      |
      +-- Impl: Actual implementation
```

### Problem Statement

How can we redesign our lower level APIs to allow composition of distance computations?

## Proposal

The solution is relatively simple and is probably a variant of some visitor pattern.
For the purposes of this demonstration, assume we have two level of distance function factories.
Level 1 dispatches between adding or multiplying two numbers `x` and `y`.
Level 2 first doubles both arguments before calling level 1.
While this is contrived, it is a close match for the more complicated problem statement outlined in the introduction.

### Existing Approach

First, the existing approach in the library is shown to demonstrate what we're working against.

```rust
// Select between adding and multiplying
enum Op {
    Add,
    Multiply,
}

// Return a function pointer implementing the requested `Op`.
fn level_1_factory(op: Op) -> fn(f32, f32) -> f32 {
    match op {
        Op::Add => |x: f32, y: f32| -> f32 { x + y },
        Op::Multiply => |x: f32, y: f32| -> f32 { x * y },
    }
}

// Wrap the function from level 1 in another functor that doubles the arguments.
fn level_2_factory(op: Op) -> Box<dyn Fn(f32, f32) -> f32> {
    let level_1 = level_1_factory(op);
    Box::new(move |x, y| level_1(2.0 * x, 2.0 * y))
}
```

Here the generated code for the closure looks like this:
```
example::level_2_factory::{{closure}}::hfb8805e5c28d1541:
        mov     rax, qword ptr [rdi]    // Load address of the `level_1` function pointer.
        addss   xmm0, xmm0              // Multiply `x` by 2.0
        addss   xmm1, xmm1              // Multiply `y` by 2.0
        jmp     rax                     // Call the `level_1` function pointer

```
Note the dynamic dispatch (`jmp`).
For completeness, here is the code for the `Add` and `Multiply` functions respectively.
```
core::ops::function::FnOnce::call_once::h59af9d1b52121d4a:
        addss   xmm0, xmm1
        ret

core::ops::function::FnOnce::call_once::h2f299cadabe46f15:
        mulss   xmm0, xmm1
        ret
```
We're paying an indirection (in this case, a tailcall jump because the inner function has the same ABI as the outer one) to run a single instruction.

### The Solution

The solution here is to use a visitor implementing "bring your own type erasure":
```rust
enum Op {
    Add,
    Multiply,
}

// Instead of return a function pointer from level 1, we visit implementations of `Level1`.
trait Level1: 'static {
    fn call(&self, x: f32, y: f32) -> f32;
}

// Internally, we use function objects (not pointers).
impl<F> Level1 for F
where
    F: Fn(f32, f32) -> f32 + 'static,
{
    fn call(&self, x: f32, y: f32) -> f32 {
        (self)(x, y)
    }
}

// Callers implement `Erase` to go from a `Level1` implementation to the final type-erased object.
trait Erase {
    // The type of the type-erased object.
    type Output;

    // Type-erase a `Level1` object.
    fn erase<F>(self, f: F) -> Self::Output
    where
        F: Level1;
}

// Implement Level 1 via visitation.
fn level_1_factory<E>(op: Op, erase: E) -> E::Output
where
    E: Erase,
{
    match op {
        Op::Add => erase.erase(|x: f32, y: f32| -> f32 { x + y }),
        Op::Multiply => erase.erase(|x: f32, y: f32| -> f32 { x * y }),
    }
}

// Wrap the function from level 1 in another functor that doubles the arguments.
fn level_2_factory(op: Op) -> Box<dyn Fn(f32, f32) -> f32> {
    struct Visit;
    impl Erase for Visit {
        type Output = Box<dyn Fn(f32, f32) -> f32>;
        fn erase<F>(self, f: F) -> Self::Output
        where
            F: Level1
        {
            // The key difference here is that we have the **concrete** type of `f` rather
            // than a type erased object (function pointer).
            Box::new(move |x, y| f.call(2.0 * x, 2.0 * y))
        }
    }

    level_1_factory(op, Visit)
}
```
With this, the level 2 implementations for `Add` and `Multiply` become
```
core::ops::function::FnOnce::call_once{{vtable.shim}}::hb25f5bff92c4657b:
        addss   xmm0, xmm0
        addss   xmm1, xmm1
        addss   xmm0, xmm1
        ret

core::ops::function::FnOnce::call_once{{vtable.shim}}::he89f81259eb002ea:
        addss   xmm0, xmm0
        addss   xmm1, xmm1
        mulss   xmm0, xmm1
        ret
```
Everything is inlined!
Further, `level_1_factory` is free to add more implementations that will automatically be fused by `level_2_factory`.

### Areas where this can be used

* `DistanceProvider`: The distance provider trait can have a bring-your-own-type-erasure interface with the current usage of magic function pointers being a default provided implementation.
* Spherical quantization distance kernels: As inherent methods on `Impl` with the `Quantizer` trait calling into the inherent methods.
* Multi-Vector distance backends.

## Trade-offs

The main trade-offs here are API complexity and compile times.
If the `level_1_factory` dispatches to many possible implementations, like the `DistanceProvider` API which dispatches across micro-architecture, metric, and length specialization, each higher level essentially redoes that work.

However, for distance functions that are called millions or billions of time in a hot loop, the extra complexity to minimize overhead is often worth it.

