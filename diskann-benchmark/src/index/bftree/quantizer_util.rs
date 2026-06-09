/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    alloc::{AllocatorError, GlobalAllocator, Poly},
    spherical::{
        iface::{self as spherical_iface, Quantizer},
        SphericalQuantizer,
    },
};
use diskann_utils::views::MatrixView;
use rand::SeedableRng;

use crate::{inputs::bftree::QuantConfig, utils::SimilarityMeasure};

fn new_quantizer<const NBITS: usize>(
    quantizer: SphericalQuantizer,
) -> Result<Poly<dyn Quantizer>, AllocatorError>
where
    spherical_iface::Impl<NBITS>: spherical_iface::Constructible + Quantizer,
{
    let imp = spherical_iface::Impl::<NBITS>::new(quantizer)?;
    diskann_quantization::poly!(Quantizer, imp, GlobalAllocator)
}

pub(super) fn build_quantizer(
    quantization: &QuantConfig,
    data: MatrixView<'_, f32>,
    distance: SimilarityMeasure,
) -> anyhow::Result<Option<Poly<dyn Quantizer>>> {
    match quantization {
        QuantConfig::None => Ok(None),
        QuantConfig::Spherical {
            seed,
            transform_kind,
            num_bits,
            pre_scale,
            ..
        } => {
            let m: diskann_vector::distance::Metric = distance.into();
            let pre_scale = match pre_scale {
                Some(v) => (*v).try_into()?,
                None => diskann_quantization::spherical::PreScale::None,
            };

            let quantizer = SphericalQuantizer::train(
                data,
                transform_kind.into(),
                m.try_into()?,
                pre_scale,
                &mut rand::rngs::StdRng::seed_from_u64(*seed),
                GlobalAllocator,
            )?;

            let poly = match num_bits.get() {
                1 => new_quantizer::<1>(quantizer)?,
                2 => new_quantizer::<2>(quantizer)?,
                4 => new_quantizer::<4>(quantizer)?,
                n => anyhow::bail!("{n} bits not supported for spherical quantization"),
            };

            Ok(Some(poly))
        }
    }
}
