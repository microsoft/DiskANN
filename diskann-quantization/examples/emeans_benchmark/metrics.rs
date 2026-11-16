/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    CompressInto, algorithms::kmeans::lloyds, multi_vector::BlockTransposed, product::BasicTable,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use std::error::Error;

fn squared_distance(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum()
}

/// Fresh final-center assignment; never use either trainer's moving/stale labels.
pub fn clustering_mse(data: rowmajor::Ref<'_, f32>, table: &BasicTable) -> f64 {
    let centers = table.view_pivots();
    let norms = |matrix: rowmajor::Ref<'_, f32>| {
        matrix
            .rows()
            .map(|r| r.iter().map(|x| x * x).sum())
            .collect::<Vec<f32>>()
    };
    let transpose = BlockTransposed::<f32, 16>::from_matrix_view(data);
    let mut labels = vec![0; data.nrows()];
    lloyds::distances_in_place(
        transpose.as_view(),
        &norms(data),
        centers,
        &norms(centers),
        &mut labels,
    );
    data.rows()
        .zip(labels)
        .map(|(row, label)| squared_distance(row, centers.row(label as usize)))
        .sum::<f64>()
        / data.nrows() as f64
}

/// Use the production BasicTable byte encoder, then score reconstruction in f64.
pub fn encode(
    data: rowmajor::Ref<'_, f32>,
    table: &BasicTable,
) -> Result<(Vec<u8>, f64), Box<dyn Error>> {
    let mut codes = vec![0; data.nrows() * table.nchunks()];
    let mut error = 0.0;
    for (row, code) in data.rows().zip(codes.chunks_mut(table.nchunks())) {
        table.compress_into(row, &mut *code)?;
        for (part, &center) in code.iter().enumerate() {
            let range = table.view_offsets().at(part);
            error += squared_distance(
                &row[range.clone()],
                &table.view_pivots().row(center as usize)[range],
            );
        }
    }
    Ok((codes, error / data.nrows() as f64))
}

fn top_k(distances: impl Iterator<Item = f64>, k: usize) -> Vec<usize> {
    let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
    for (index, distance) in distances.enumerate() {
        if best.len() == k
            && best
                .last()
                .is_some_and(|&(d, i)| (distance, index) >= (d, i))
        {
            continue;
        }
        let pos = best.partition_point(|&(d, i)| (d, i) <= (distance, index));
        best.insert(pos, (distance, index));
        if best.len() > k {
            best.pop();
        }
    }
    best.into_iter().map(|(_, index)| index).collect()
}

pub fn ground_truth(
    database: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: usize,
) -> Vec<Vec<usize>> {
    queries
        .rows()
        .map(|query| top_k(database.rows().map(|row| squared_distance(query, row)), k))
        .collect()
}

fn adc_lookup(query: &[f32], table: &BasicTable) -> Vec<f64> {
    let mut lookup = vec![0.0; table.nchunks() * table.ncenters()];
    for part in 0..table.nchunks() {
        let range = table.view_offsets().at(part);
        for (center, row) in table.view_pivots().rows().enumerate() {
            lookup[part * table.ncenters() + center] =
                squared_distance(&query[range.clone()], &row[range.clone()]);
        }
    }
    lookup
}

pub fn recall(
    codes: &[u8],
    queries: rowmajor::Ref<'_, f32>,
    table: &BasicTable,
    truth: &[Vec<usize>],
) -> f64 {
    let mut found = 0;
    let mut total = 0;
    for (query, expected) in queries.rows().zip(truth) {
        let lookup = adc_lookup(query, table);
        let nearest = top_k(
            codes.chunks_exact(table.nchunks()).map(|code| {
                code.iter()
                    .enumerate()
                    .map(|(part, &center)| lookup[part * table.ncenters() + center as usize])
                    .sum()
            }),
            expected.len(),
        );
        found += nearest
            .iter()
            .filter(|index| expected.contains(index))
            .count();
        total += expected.len();
    }
    found as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_quantization::views::ChunkOffsets;
    use std::num::NonZeroUsize;

    #[test]
    fn reconstruction_and_adc_use_the_query_without_quantizing_it() -> Result<(), Box<dyn Error>> {
        let table = BasicTable::new(
            rowmajor::Owned::from_fn(2, 2, |rc| rc.row as f32 * 10.0),
            ChunkOffsets::partition(
                NonZeroUsize::new(2).ok_or("test dimension")?,
                NonZeroUsize::new(2).ok_or("test chunks")?,
            )?,
        )?;
        let data = rowmajor::Ref::try_from_data(&[0.0, 0.0, 1.0, 1.0, 9.0, 9.0], 3, 2)?;
        let query = rowmajor::Ref::try_from_data(&[8.0, 8.0], 1, 2)?;
        let (codes, mse) = encode(data, &table)?;
        assert_eq!(codes, [0, 0, 0, 0, 1, 1]);
        assert!((mse - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(adc_lookup(query.row(0), &table), [64.0, 4.0, 64.0, 4.0]);
        let truth = ground_truth(data, query, 2);
        assert_eq!(truth, [vec![2, 1]]);
        assert_eq!(recall(&codes, query, &table, &truth), 0.5);
        Ok(())
    }

    #[test]
    fn final_clustering_loss_and_distance_ties_are_explicit() -> Result<(), Box<dyn Error>> {
        let table = BasicTable::new(
            rowmajor::Owned::from_fn(2, 2, |rc| rc.row as f32 * 10.0),
            ChunkOffsets::partition(
                NonZeroUsize::new(2).ok_or("test dimension")?,
                NonZeroUsize::new(1).ok_or("test chunks")?,
            )?,
        )?;
        let data = rowmajor::Ref::try_from_data(&[0.0, 0.0, 1.0, 1.0, 9.0, 9.0], 3, 2)?;
        assert!((clustering_mse(data, &table) - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(top_k([2.0, 1.0, 1.0, 0.0].into_iter(), 3), [3, 1, 2]);
        Ok(())
    }
}
