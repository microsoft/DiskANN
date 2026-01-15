/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[derive(Debug, Clone)]
pub struct DegreeStats {
    pub max_degree: u32,
    pub avg_degree: f32,
    pub min_degree: u32,
    pub cnt_less_than_two: usize, // Number of vertices with degree less than 2
}
