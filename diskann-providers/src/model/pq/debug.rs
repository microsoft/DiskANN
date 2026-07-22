/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::IntoUsize;
use diskann_utils::views;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

pub struct MismatchRecord {
    pub row: usize,
    pub chunk: usize,
    pub a_assignment: usize,
    pub a_pivot: Vec<f32>,
    pub b_assignment: usize,
    pub b_pivot: Vec<f32>,
    pub data: Vec<f32>,
    pub center: Vec<f32>,
    pub squared_l2_a: f32,
    pub squared_l2_b: f32,
}

impl std::fmt::Display for MismatchRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "mismatch on row {} and chunk {}", self.row, self.chunk)?;
        writeln!(
            f,
            "argument A had assignment {} but B had assignment {}",
            self.a_assignment, self.b_assignment
        )?;
        writeln!(f, "data = {:?}", self.data)?;
        writeln!(f, "center = {:?}", self.center)?;
        writeln!(f, "pivot_a = {:?}", self.a_pivot)?;
        writeln!(f, "pivot_b = {:?}", self.b_pivot)?;
        writeln!(f, "distance from a = {}", self.squared_l2_a)?;
        writeln!(f, "distance from b = {}", self.squared_l2_b)
    }
}

/// NOT A PRODUCTION READY FUNCTION
///
/// Check the two PQ compressions "a" and "b" for equality.
///
/// For all entries that are not equal, construct a `MismatchRecord` for the row and chunk
/// of the mismatch, recording the details for further diagnostics.
///
/// This is not a production ready function as it does not perform extensive error checking
/// on the sizes of the provided arguments, but can be helpful for writing test routines
/// and as such is still marked as public.
pub fn compare_pq<T, U>(
    data: views::MatrixView<'_, T>,
    schema: diskann_quantization::views::ChunkOffsetsView<'_>,
    pivots: views::MatrixView<'_, f32>,
    center: &[f32],
    a: views::MatrixView<'_, U>,
    b: views::MatrixView<'_, U>,
) -> Vec<MismatchRecord>
where
    T: Copy + Into<f32>,
    U: Copy + IntoUsize,
{
    std::iter::zip(a.row_iter(), b.row_iter())
        .enumerate()
        .flat_map(|(row, (a_row, b_row))| {
            std::iter::zip(a_row.iter(), b_row.iter())
                .enumerate()
                .filter_map(move |(chunk, (a, b))| {
                    let a: usize = a.into_usize();
                    let b: usize = b.into_usize();
                    // This is a match - nothing to do.
                    if a == b {
                        return None;
                    }

                    let range = schema.at(chunk);

                    // There is a mismatch.
                    // Time to go to work.
                    // Get the pivot A was assigned to.
                    let source_data: Vec<f32> = data.row(row)[range.clone()]
                        .iter()
                        .map(|&x| x.into())
                        .collect();
                    let center = center[range.clone()].to_vec();
                    let a_pivot = pivots.row(a)[range.clone()].to_vec();
                    let b_pivot = pivots.row(b)[range.clone()].to_vec();

                    // Compute the L2 distance between the source data without the center and
                    // the pivots selected by A and B.
                    let source_data_compensated: Vec<f32> =
                        std::iter::zip(source_data.iter(), center.iter())
                            .map(|(s, c)| s - c)
                            .collect();

                    let squared_l2_a =
                        SquaredL2::evaluate(source_data_compensated.as_slice(), a_pivot.as_slice());

                    let squared_l2_b =
                        SquaredL2::evaluate(source_data_compensated.as_slice(), b_pivot.as_slice());

                    Some(MismatchRecord {
                        row,
                        chunk,
                        a_assignment: a,
                        a_pivot,
                        b_assignment: b,
                        b_pivot,
                        data: source_data,
                        center,
                        squared_l2_a,
                        squared_l2_b,
                    })
                })
        })
        .collect()
}
