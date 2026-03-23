/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::lazy_format;
use diskann_vector::{
    Norm, PureDistanceFunction,
    distance::{InnerProduct, SquaredL2},
    norm::FastL2NormSquared,
};
use rand::{distr::Distribution, rngs::StdRng};
use rand_distr::StandardNormal;

use super::TransformFailed;

use crate::test_util::Check;

pub(super) trait Transformer {
    fn input_dim_(&self) -> usize;
    fn output_dim_(&self) -> usize;
    fn transform_into_(&self, dst: &mut [f32], src: &[f32]) -> Result<(), TransformFailed>;
}

macro_rules! delegate_transformer {
    ($T:ty) => {
        impl $crate::algorithms::transforms::test_utils::Transformer for $T {
            fn input_dim_(&self) -> usize {
                <$T>::input_dim(self)
            }
            fn output_dim_(&self) -> usize {
                <$T>::output_dim(self)
            }
            fn transform_into_(
                &self,
                dst: &mut [f32],
                src: &[f32]
            ) -> Result<(), $crate::algorithms::transforms::TransformFailed> {
                <$T>::transform_into(self, dst, src, $crate::alloc::ScopedAllocator::global())
            }
        }
    };
    ($T:ty, $($Ts:ty),+) => {
        delegate!($T);
        $(delegate!($Ts);)+
    };
}

pub(super) use delegate_transformer;

pub(super) struct IO<'a> {
    pub(super) input0: &'a [f32],
    pub(super) input1: &'a [f32],
    pub(super) output0: &'a [f32],
    pub(super) output1: &'a [f32],
}

pub(super) fn test_transform(
    transformer: &dyn Transformer,
    num_trials: usize,
    checker: &mut dyn FnMut(IO<'_>, &dyn std::fmt::Display),
    rng: &mut StdRng,
    context: &dyn std::fmt::Display,
) {
    let input_dim = transformer.input_dim_();
    let output_dim = transformer.output_dim_();

    // Errors on output dimension
    {
        // Check error handling.
        let good_input = vec![0.0f32; input_dim];
        let mut bad_output = vec![0.0f32; output_dim + 1];

        let err = transformer
            .transform_into_(&mut bad_output, &good_input)
            .unwrap_err();

        let expected = TransformFailed::DestinationMismatch {
            expected: output_dim,
            found: output_dim + 1,
        };
        assert_eq!(err, expected);

        let err = transformer
            .transform_into_(&mut [], &good_input)
            .unwrap_err();

        let expected = TransformFailed::DestinationMismatch {
            expected: output_dim,
            found: 0,
        };
        assert_eq!(err, expected);
    }

    // Errors on input dimension
    {
        // Check error handling.
        let bad_input = vec![0.0f32; input_dim + 1];
        let mut good_output = vec![0.0f32; output_dim];

        let err = transformer
            .transform_into_(&mut good_output, &bad_input)
            .unwrap_err();

        let expected = TransformFailed::SourceMismatch {
            expected: input_dim,
            found: input_dim + 1,
        };
        assert_eq!(err, expected);

        let err = transformer
            .transform_into_(&mut good_output, &[])
            .unwrap_err();

        let expected = TransformFailed::SourceMismatch {
            expected: input_dim,
            found: 0,
        };
        assert_eq!(err, expected);
    }

    let mut input0 = vec![0.0f32; input_dim];
    let mut input1 = vec![0.0f32; input_dim];
    let mut output0 = vec![0.0f32; output_dim];
    let mut output1 = vec![0.0f32; output_dim];

    let populate = |v: &mut [f32], rng: &mut StdRng| {
        v.iter_mut()
            .for_each(|i| *i = StandardNormal {}.sample(rng));
    };

    for trial in 0..num_trials {
        populate(&mut input0, rng);
        populate(&mut input1, rng);

        transformer.transform_into_(&mut output0, &input0).unwrap();
        transformer.transform_into_(&mut output1, &input1).unwrap();

        checker(
            IO {
                input0: &input0,
                input1: &input1,
                output0: &output0,
                output1: &output1,
            },
            &lazy_format!("{}, trial {} of {}", context, trial, num_trials),
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ErrorSetup {
    /// The error bound for the norm.
    pub(super) norm: Check,

    /// The error bound for L2
    pub(super) l2: Check,

    /// The error bound for inner product.
    pub(super) ip: Check,
}

pub(super) fn check_errors(io: IO<'_>, context: &dyn std::fmt::Display, errors: &ErrorSetup) {
    // Check Norms
    let input_norm0 = FastL2NormSquared.evaluate(io.input0);
    let output_norm0 = FastL2NormSquared.evaluate(io.output0);

    let input_norm1 = FastL2NormSquared.evaluate(io.input1);
    let output_norm1 = FastL2NormSquared.evaluate(io.output1);

    if let Err(err) = errors.norm.check(output_norm0, input_norm0) {
        panic!("Norm check failed: {} -- {}", err, context);
    }

    if let Err(err) = errors.norm.check(output_norm1, input_norm1) {
        panic!("Norm check failed: {} -- {}", err, context);
    }

    // Check L2
    {
        let l2_input: f32 = SquaredL2::evaluate(io.input0, io.input1);
        let l2_output: f32 = SquaredL2::evaluate(io.output0, io.output1);

        if let Err(err) = errors.l2.check(l2_output, l2_input) {
            panic!("L2 check failed: {} -- {}", err, context);
        }
    }

    // Check Inner Product
    {
        let ip_input: f32 = InnerProduct::evaluate(io.input0, io.input1);
        let ip_output: f32 = InnerProduct::evaluate(io.output0, io.output1);

        if let Err(err) = errors.ip.check(ip_output, ip_input) {
            panic!("IP check failed: {} -- {}", err, context);
        }
    }
}
