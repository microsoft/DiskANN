/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    alloc::{Allocator, AllocatorError, BumpAllocator, CompoundError, Poly},
    num::PowerOfTwo,
    poly,
};

trait DoTheThing {
    fn do_the_thing(&self) -> String;
}

#[derive(Debug, Clone, Copy)]
enum TransformKind {
    Hadamard,
    Null,
}

enum Transform<A>
where
    A: Allocator,
{
    Hadamard { _signs: Poly<[u32], A> },
    Null,
}

#[derive(Debug, Clone, Copy)]
struct DimKind {
    dim: usize,
    kind: TransformKind,
}

/////////////////////////
// Quantization Scheme //
/////////////////////////

struct Quantizer<A>
where
    A: Allocator,
{
    centroid: Poly<[f32], A>,
    _scale: f32,
    transform: Transform<A>,
}

impl<A> Quantizer<A>
where
    A: Allocator + Clone,
{
    fn new(
        dim_kind: DimKind,
        allocator: A,
    ) -> Result<Poly<Self, A>, CompoundError<AllocatorError>> {
        Poly::new_with(
            |allocator| {
                let centroid = Poly::from_iter((0..dim_kind.dim).map(|_| 0.0), allocator.clone())?;
                let transform = match dim_kind.kind {
                    TransformKind::Hadamard => Transform::Hadamard {
                        _signs: Poly::from_iter((0..dim_kind.dim).map(|_| 0), allocator.clone())?,
                    },
                    TransformKind::Null => Transform::Null,
                };

                Ok(Self {
                    centroid,
                    _scale: 0.0,
                    transform,
                })
            },
            allocator,
        )
    }
}

impl<A> DoTheThing for Quantizer<A>
where
    A: Allocator,
{
    fn do_the_thing(&self) -> String {
        "foo".into()
    }
}

///////////
// Tests //
///////////

#[test]
fn miri_q1_no_transform() {
    let dim_kind = DimKind {
        dim: 128,
        kind: TransformKind::Null,
    };

    let allocator = BumpAllocator::new(4096, PowerOfTwo::new(4096).unwrap()).unwrap();
    let object = Quantizer::new(dim_kind, allocator.clone()).unwrap();

    // Assert that the pointer for the object is the same as the base pointer for the allocator.
    let base = allocator.as_ptr();
    assert_eq!(Poly::as_ptr(&object).cast::<u8>(), base);

    // The data for the next allocation comes next in the page.
    //
    // SAFETY:
    assert_eq!(
        object.centroid.as_ptr().cast::<u8>(),
        base.wrapping_add(
            std::mem::size_of::<Quantizer<BumpAllocator>>()
                .next_multiple_of(std::mem::align_of::<f32>()),
        )
    );

    assert!(matches!(object.transform, Transform::Null));
}

#[test]
fn miri_q1_transform() {
    let dim_kind = DimKind {
        dim: 128,
        kind: TransformKind::Hadamard,
    };

    let allocator = BumpAllocator::new(4096, PowerOfTwo::new(4096).unwrap()).unwrap();
    let object = Quantizer::new(dim_kind, allocator.clone()).unwrap();

    // Assert that the pointer for the object is the same as the base pointer for the allocator.
    let base = allocator.as_ptr();
    assert_eq!(Poly::as_ptr(&object).cast::<u8>(), base);

    // The data for the next allocation comes next in the page.
    assert_eq!(
        object.centroid.as_ptr().cast::<u8>(),
        base.wrapping_add(
            std::mem::size_of::<Quantizer<BumpAllocator>>()
                .next_multiple_of(std::mem::align_of::<f32>()),
        )
    );
}

#[test]
fn miri_trait_object_as_base() {
    let dim_kind = DimKind {
        dim: 128,
        kind: TransformKind::Hadamard,
    };

    let allocator = BumpAllocator::new(4096, PowerOfTwo::new(4096).unwrap()).unwrap();
    let poly: Poly<Poly<dyn DoTheThing, BumpAllocator>, BumpAllocator> = Poly::new_with(
        |allocator| -> Result<_, std::alloc::LayoutError> {
            let object = Quantizer::new(dim_kind, allocator).unwrap();
            Ok(poly!(DoTheThing, object))
        },
        allocator.clone(),
    )
    .unwrap();

    // Demonstrate that the object pointer is working properly.
    let x: &dyn DoTheThing = &**poly;
    assert_eq!(x.do_the_thing(), "foo");

    // The base object is at the base of the allocator.
    {
        let base = allocator.as_ptr();
        assert_eq!(Poly::as_ptr(&poly).cast::<u8>(), base);
    }

    // We can use pointer directly as a trait object.
    {
        let ptr = Poly::as_ptr(&poly);

        // SAFETY: `ptr` was obtained from a valid object that is still alive.
        let object = unsafe { &*ptr };
        assert_eq!(object.do_the_thing(), "foo");
    }
}
