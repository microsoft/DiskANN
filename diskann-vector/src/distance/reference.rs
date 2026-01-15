/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{distance::Metric, Half, MathematicalValue, SimilarityScore};

////////////////////////
// Reference Provider //
////////////////////////

pub(crate) trait ReferenceProvider<L, R = L> {
    fn reference_implementation(metric: Metric) -> fn(&[L], &[R]) -> SimilarityScore<f32>;
}

impl ReferenceProvider<f32> for f32 {
    fn reference_implementation(metric: Metric) -> fn(&[f32], &[f32]) -> SimilarityScore<f32> {
        match metric {
            Metric::L2 => reference_squared_l2_f32_similarity,
            Metric::InnerProduct => reference_innerproduct_f32_similarity,
            Metric::Cosine => reference_cosine_f32_similarity,
            Metric::CosineNormalized => reference_cosine_normalized_f32_similarity,
        }
    }
}

impl ReferenceProvider<Half> for Half {
    fn reference_implementation(metric: Metric) -> fn(&[Half], &[Half]) -> SimilarityScore<f32> {
        match metric {
            Metric::L2 => reference_squared_l2_f16_similarity,
            Metric::InnerProduct => reference_innerproduct_f16_similarity,
            Metric::Cosine => reference_cosine_f16_similarity,
            Metric::CosineNormalized => reference_cosine_normalized_f16_similarity,
        }
    }
}

impl ReferenceProvider<i8> for i8 {
    // NOTE: `CosineNormalized` behaves like `Cosine`.
    fn reference_implementation(metric: Metric) -> fn(&[i8], &[i8]) -> SimilarityScore<f32> {
        match metric {
            Metric::L2 => reference_squared_l2_i8_similarity,
            Metric::InnerProduct => reference_innerproduct_i8_similarity,
            Metric::Cosine => reference_cosine_i8_similarity,
            Metric::CosineNormalized => reference_cosine_i8_similarity,
        }
    }
}

impl ReferenceProvider<u8> for u8 {
    // NOTE: `CosineNormalized` behaves like `Cosine`.
    fn reference_implementation(metric: Metric) -> fn(&[u8], &[u8]) -> SimilarityScore<f32> {
        match metric {
            Metric::L2 => reference_squared_l2_u8_similarity,
            Metric::InnerProduct => reference_innerproduct_u8_similarity,
            Metric::Cosine => reference_cosine_u8_similarity,
            Metric::CosineNormalized => reference_cosine_u8_similarity,
        }
    }
}

///////////////
// SquaredL2 //
///////////////

// Mathematical
pub fn reference_squared_l2_i8_mathematical(x: &[i8], y: &[i8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter())
        .map(|(&a, &b)| {
            let a: i32 = a.into();
            let b: i32 = b.into();
            let diff = a - b;
            diff * diff
        })
        .sum::<i32>() as f32;

    MathematicalValue::new(r)
}

pub fn reference_squared_l2_u8_mathematical(x: &[u8], y: &[u8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter())
        .map(|(&a, &b)| {
            let a: i32 = a.into();
            let b: i32 = b.into();
            let diff = a - b;
            diff * diff
        })
        .sum::<i32>() as f32;

    MathematicalValue::new(r)
}

pub fn reference_squared_l2_f16_mathematical(x: &[Half], y: &[Half]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let a: f32 = diskann_wide::cast_f16_to_f32(a);
        let b: f32 = diskann_wide::cast_f16_to_f32(b);
        let diff = a - b;
        diff.mul_add(diff, acc)
    });

    MathematicalValue::new(r)
}

pub fn reference_squared_l2_f32_mathematical(x: &[f32], y: &[f32]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let diff = a - b;
        diff.mul_add(diff, acc)
    });

    MathematicalValue::new(r)
}

pub fn reference_l1_f16_mathematical(x: &[Half]) -> MathematicalValue<f32> {
    let sum: f32 = x.iter().fold(0.0f32, |acc, &h| {
        let v: f32 = diskann_wide::cast_f16_to_f32(h);
        acc + v.abs()
    });
    MathematicalValue::new(sum)
}

pub fn reference_l1_f32_mathematical(x: &[f32]) -> MathematicalValue<f32> {
    let sum: f32 = x.iter().fold(0.0f32, |acc, &h| acc + h.abs());
    MathematicalValue::new(sum)
}

pub fn reference_linf_f16_mathematical(x: &[Half]) -> MathematicalValue<f32> {
    let mut m = 0.0f32;
    for &h in x {
        let v: f32 = diskann_wide::cast_f16_to_f32(h);
        m = m.max(v.abs());
    }
    MathematicalValue::new(m)
}

pub fn reference_linf_f32_mathematical(x: &[f32]) -> MathematicalValue<f32> {
    let mut m = 0.0f32;
    for &h in x {
        m = m.max(h.abs());
    }
    MathematicalValue::new(m)
}

pub fn reference_squared_l2_f32xf16_mathematical(x: &[f32], y: &[Half]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let b: f32 = diskann_wide::cast_f16_to_f32(b);
        let diff = a - b;
        diff.mul_add(diff, acc)
    });

    MathematicalValue::new(r)
}

// Similarity
pub fn reference_squared_l2_i8_similarity(x: &[i8], y: &[i8]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_squared_l2_i8_mathematical(x, y).into_inner())
}

pub fn reference_squared_l2_u8_similarity(x: &[u8], y: &[u8]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_squared_l2_u8_mathematical(x, y).into_inner())
}

pub fn reference_squared_l2_f16_similarity(x: &[Half], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_squared_l2_f16_mathematical(x, y).into_inner())
}

pub fn reference_squared_l2_f32_similarity(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_squared_l2_f32_mathematical(x, y).into_inner())
}

pub fn reference_squared_l2_f32xf16_similarity(x: &[f32], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_squared_l2_f32xf16_mathematical(x, y).into_inner())
}

////////
// L2 //
////////

// Mathematical
pub fn reference_l2_i8_mathematical(x: &[i8], y: &[i8]) -> MathematicalValue<f32> {
    MathematicalValue::new(
        reference_squared_l2_i8_mathematical(x, y)
            .into_inner()
            .sqrt(),
    )
}

pub fn reference_l2_u8_mathematical(x: &[u8], y: &[u8]) -> MathematicalValue<f32> {
    MathematicalValue::new(
        reference_squared_l2_u8_mathematical(x, y)
            .into_inner()
            .sqrt(),
    )
}

pub fn reference_l2_f16_mathematical(x: &[Half], y: &[Half]) -> MathematicalValue<f32> {
    MathematicalValue::new(
        reference_squared_l2_f16_mathematical(x, y)
            .into_inner()
            .sqrt(),
    )
}

pub fn reference_l2_f32_mathematical(x: &[f32], y: &[f32]) -> MathematicalValue<f32> {
    MathematicalValue::new(
        reference_squared_l2_f32_mathematical(x, y)
            .into_inner()
            .sqrt(),
    )
}

pub fn reference_l2_f32xf16_mathematical(x: &[f32], y: &[Half]) -> MathematicalValue<f32> {
    MathematicalValue::new(
        reference_squared_l2_f32xf16_mathematical(x, y)
            .into_inner()
            .sqrt(),
    )
}

// Similarity
pub fn reference_l2_i8_similarity(x: &[i8], y: &[i8]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_l2_i8_mathematical(x, y).into_inner())
}

pub fn reference_l2_u8_similarity(x: &[u8], y: &[u8]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_l2_u8_mathematical(x, y).into_inner())
}

pub fn reference_l2_f16_similarity(x: &[Half], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_l2_f16_mathematical(x, y).into_inner())
}

pub fn reference_l2_f32_similarity(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_l2_f32_mathematical(x, y).into_inner())
}

pub fn reference_l2_f32xf16_similarity(x: &[f32], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(reference_l2_f32xf16_mathematical(x, y).into_inner())
}

///////////////////
// Inner Product //
///////////////////

// Mathematical
pub fn reference_innerproduct_i8_mathematical(x: &[i8], y: &[i8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter())
        .map(|(&a, &b)| {
            let a: i32 = a.into();
            let b: i32 = b.into();
            a * b
        })
        .sum::<i32>() as f32;

    MathematicalValue::new(r)
}

pub fn reference_innerproduct_u8_mathematical(x: &[u8], y: &[u8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter())
        .map(|(&a, &b)| {
            let a: i32 = a.into();
            let b: i32 = b.into();
            a * b
        })
        .sum::<i32>() as f32;

    MathematicalValue::new(r)
}

pub fn reference_innerproduct_f16_mathematical(x: &[Half], y: &[Half]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let a: f32 = diskann_wide::cast_f16_to_f32(a);
        let b: f32 = diskann_wide::cast_f16_to_f32(b);
        a.mul_add(b, acc)
    });

    MathematicalValue::new(r)
}

pub fn reference_innerproduct_f32_mathematical(x: &[f32], y: &[f32]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| a.mul_add(b, acc));

    MathematicalValue::new(r)
}

pub fn reference_innerproduct_f32xf16_mathematical(
    x: &[f32],
    y: &[Half],
) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let b: f32 = b.into();
        a.mul_add(b, acc)
    });

    MathematicalValue::new(r)
}

// Similarity
pub fn reference_innerproduct_i8_similarity(x: &[i8], y: &[i8]) -> SimilarityScore<f32> {
    SimilarityScore::new(-reference_innerproduct_i8_mathematical(x, y).into_inner())
}

pub fn reference_innerproduct_u8_similarity(x: &[u8], y: &[u8]) -> SimilarityScore<f32> {
    SimilarityScore::new(-reference_innerproduct_u8_mathematical(x, y).into_inner())
}

pub fn reference_innerproduct_f16_similarity(x: &[Half], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(-reference_innerproduct_f16_mathematical(x, y).into_inner())
}

pub fn reference_innerproduct_f32_similarity(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
    SimilarityScore::new(-reference_innerproduct_f32_mathematical(x, y).into_inner())
}

pub fn reference_innerproduct_f32xf16_similarity(x: &[f32], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(-reference_innerproduct_f32xf16_mathematical(x, y).into_inner())
}

////////////
// Cosine //
////////////

#[derive(Default)]
struct XY<T> {
    xnorm: T,
    ynorm: T,
    xy: T,
}

// Mathematical
pub fn reference_cosine_i8_mathematical(x: &[i8], y: &[i8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: XY<i32> =
        std::iter::zip(x.iter(), y.iter()).fold(XY::<i32>::default(), |acc, (&vx, &vy)| {
            let vx: i32 = vx.into();
            let vy: i32 = vy.into();
            XY {
                xnorm: acc.xnorm + vx * vx,
                ynorm: acc.ynorm + vy * vy,
                xy: acc.xy + vx * vy,
            }
        });

    if r.xnorm == 0 || r.ynorm == 0 {
        return MathematicalValue::new(0.0);
    }

    MathematicalValue::new(
        (r.xy as f32 / ((r.xnorm as f32).sqrt() * (r.ynorm as f32).sqrt())).clamp(-1.0, 1.0),
    )
}

pub fn reference_cosine_u8_mathematical(x: &[u8], y: &[u8]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: XY<i32> =
        std::iter::zip(x.iter(), y.iter()).fold(XY::<i32>::default(), |acc, (&vx, &vy)| {
            let vx: i32 = vx.into();
            let vy: i32 = vy.into();
            XY {
                xnorm: acc.xnorm + vx * vx,
                ynorm: acc.ynorm + vy * vy,
                xy: acc.xy + vx * vy,
            }
        });

    if r.xnorm == 0 || r.ynorm == 0 {
        return MathematicalValue::new(0.0);
    }

    MathematicalValue::new(
        (r.xy as f32 / ((r.xnorm as f32).sqrt() * (r.ynorm as f32).sqrt())).clamp(-1.0, 1.0),
    )
}

pub fn reference_cosine_f16_mathematical(x: &[Half], y: &[Half]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: XY<f32> =
        std::iter::zip(x.iter(), y.iter()).fold(XY::<f32>::default(), |acc, (&vx, &vy)| {
            let vx: f32 = diskann_wide::cast_f16_to_f32(vx);
            let vy: f32 = diskann_wide::cast_f16_to_f32(vy);
            XY {
                xnorm: vx.mul_add(vx, acc.xnorm),
                ynorm: vy.mul_add(vy, acc.ynorm),
                xy: vx.mul_add(vy, acc.xy),
            }
        });

    if r.xnorm < f32::MIN_POSITIVE || r.ynorm < f32::MIN_POSITIVE {
        return MathematicalValue::new(0.0);
    }

    MathematicalValue::new((r.xy / (r.xnorm.sqrt() * r.ynorm.sqrt())).clamp(-1.0, 1.0))
}

pub fn reference_cosine_f32_mathematical(x: &[f32], y: &[f32]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: XY<f32> =
        std::iter::zip(x.iter(), y.iter()).fold(XY::<f32>::default(), |acc, (&vx, &vy)| XY {
            xnorm: vx.mul_add(vx, acc.xnorm),
            ynorm: vy.mul_add(vy, acc.ynorm),
            xy: vx.mul_add(vy, acc.xy),
        });

    if r.xnorm < f32::MIN_POSITIVE || r.ynorm < f32::MIN_POSITIVE {
        return MathematicalValue::new(0.0);
    }

    MathematicalValue::new((r.xy / (r.xnorm.sqrt() * r.ynorm.sqrt())).clamp(-1.0, 1.0))
}

pub fn reference_cosine_f32xf16_mathematical(x: &[f32], y: &[Half]) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: XY<f32> =
        std::iter::zip(x.iter(), y.iter()).fold(XY::<f32>::default(), |acc, (&vx, &vy)| {
            let vy: f32 = vy.into();
            XY {
                xnorm: vx.mul_add(vx, acc.xnorm),
                ynorm: vy.mul_add(vy, acc.ynorm),
                xy: vx.mul_add(vy, acc.xy),
            }
        });

    if r.xnorm < f32::MIN_POSITIVE || r.ynorm < f32::MIN_POSITIVE {
        return MathematicalValue::new(0.0);
    }

    MathematicalValue::new((r.xy / (r.xnorm.sqrt() * r.ynorm.sqrt())).clamp(-1.0, 1.0))
}

// Similarity
pub fn reference_cosine_i8_similarity(x: &[i8], y: &[i8]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_i8_mathematical(x, y).into_inner())
}

pub fn reference_cosine_u8_similarity(x: &[u8], y: &[u8]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_u8_mathematical(x, y).into_inner())
}

pub fn reference_cosine_f16_similarity(x: &[Half], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_f16_mathematical(x, y).into_inner())
}

pub fn reference_cosine_f32_similarity(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_f32_mathematical(x, y).into_inner())
}

pub fn reference_cosine_f32xf16_similarity(x: &[f32], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_f32xf16_mathematical(x, y).into_inner())
}

//////////////////////
// CosineNormalized //
//////////////////////

// Mathematical
pub fn reference_cosine_normalized_f16_mathematical(
    x: &[Half],
    y: &[Half],
) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let a: f32 = a.into();
        let b: f32 = b.into();
        a.mul_add(b, acc)
    });

    MathematicalValue::new(r)
}

pub fn reference_cosine_normalized_f32_mathematical(
    x: &[f32],
    y: &[f32],
) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| a.mul_add(b, acc));

    MathematicalValue::new(r)
}

pub fn reference_cosine_normalized_f32xf16_mathematical(
    x: &[f32],
    y: &[Half],
) -> MathematicalValue<f32> {
    assert_eq!(x.len(), y.len());
    let r: f32 = std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
        let b: f32 = b.into();
        a.mul_add(b, acc)
    });

    MathematicalValue::new(r)
}

// Similarity
pub fn reference_cosine_normalized_f16_similarity(x: &[Half], y: &[Half]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_normalized_f16_mathematical(x, y).into_inner())
}

pub fn reference_cosine_normalized_f32_similarity(x: &[f32], y: &[f32]) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_normalized_f32_mathematical(x, y).into_inner())
}

pub fn reference_cosine_normalized_f32xf16_similarity(
    x: &[f32],
    y: &[Half],
) -> SimilarityScore<f32> {
    SimilarityScore::new(1.0 - reference_cosine_normalized_f32xf16_mathematical(x, y).into_inner())
}
