/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::graph::AdjacencyList;

pub fn generate_1d_grid_adj_list(grid_size: u32) -> Vec<AdjacencyList<u32>> {
    let mut adj_lists: Vec<AdjacencyList<u32>> = Vec::with_capacity(grid_size as usize);

    for i in 0..grid_size {
        let mut adj_list = AdjacencyList::with_capacity(2);
        if i > 0 {
            adj_list.push(i - 1);
        }
        if i < grid_size - 1 {
            adj_list.push(i + 1);
        }
        adj_lists.push(adj_list);
    }
    adj_lists
}

pub fn generate_1d_grid_vectors_f32(grid_size: u32) -> Vec<Vec<f32>> {
    (0..grid_size).map(|i| vec![i as f32]).collect()
}

pub fn generate_1d_grid_vectors_i8(grid_size: i8) -> Vec<Vec<i8>> {
    (0..grid_size).map(|i| vec![i]).collect()
}

pub fn generate_1d_grid_vectors_u8(grid_size: u8) -> Vec<Vec<u8>> {
    (0..grid_size).map(|i| vec![i]).collect()
}

pub fn map_ijk_to_grid(i: u32, j: u32, k: u32, grid_size: u32) -> u32 {
    i * grid_size * grid_size + j * grid_size + k
}

pub fn genererate_3d_grid_adj_list(grid_size: u32) -> Vec<AdjacencyList<u32>> {
    let mut adj_lists: Vec<AdjacencyList<u32>> =
        Vec::with_capacity(grid_size as usize * grid_size as usize * grid_size as usize);
    for i in 0..grid_size {
        for j in 0..grid_size {
            for k in 0..grid_size {
                let mut adj_list = AdjacencyList::new();
                if i > 0 {
                    adj_list.push(map_ijk_to_grid(i - 1, j, k, grid_size));
                }
                if i < grid_size - 1 {
                    adj_list.push(map_ijk_to_grid(i + 1, j, k, grid_size));
                }
                if j > 0 {
                    adj_list.push(map_ijk_to_grid(i, j - 1, k, grid_size));
                }
                if j < grid_size - 1 {
                    adj_list.push(map_ijk_to_grid(i, j + 1, k, grid_size));
                }
                if k > 0 {
                    adj_list.push(map_ijk_to_grid(i, j, k - 1, grid_size));
                }
                if k < grid_size - 1 {
                    adj_list.push(map_ijk_to_grid(i, j, k + 1, grid_size));
                }
                adj_lists.push(adj_list);
            }
        }
    }
    adj_lists
}

pub fn generate_3d_grid_vectors_f32(grid_size: u32) -> Vec<Vec<f32>> {
    (0..grid_size)
        .flat_map(|i| {
            (0..grid_size)
                .flat_map(move |j| (0..grid_size).map(move |k| vec![i as f32, j as f32, k as f32]))
        })
        .collect()
}

pub fn generate_3d_grid_vectors_i8(grid_size: i8) -> Vec<Vec<i8>> {
    (0..grid_size)
        .flat_map(|i| (0..grid_size).flat_map(move |j| (0..grid_size).map(move |k| vec![i, j, k])))
        .collect()
}

pub fn generate_3d_grid_vectors_u8(grid_size: u8) -> Vec<Vec<u8>> {
    (0..grid_size)
        .flat_map(|i| (0..grid_size).flat_map(move |j| (0..grid_size).map(move |k| vec![i, j, k])))
        .collect()
}

pub fn map_ijkl_to_grid(i: u32, j: u32, k: u32, l: u32, grid_size: u32) -> u32 {
    i * grid_size * grid_size * grid_size + j * grid_size * grid_size + k * grid_size + l
}

pub fn generate_4d_grid_adj_list(grid_size: u32) -> Vec<AdjacencyList<u32>> {
    let mut adj_lists: Vec<AdjacencyList<u32>> = Vec::with_capacity(
        grid_size as usize * grid_size as usize * grid_size as usize * grid_size as usize,
    );
    for i in 0..grid_size {
        for j in 0..grid_size {
            for k in 0..grid_size {
                for l in 0..grid_size {
                    let mut adj_list = AdjacencyList::new();
                    if i > 0 {
                        adj_list.push(map_ijkl_to_grid(i - 1, j, k, l, grid_size));
                    }
                    if i < grid_size - 1 {
                        adj_list.push(map_ijkl_to_grid(i + 1, j, k, l, grid_size));
                    }
                    if j > 0 {
                        adj_list.push(map_ijkl_to_grid(i, j - 1, k, l, grid_size));
                    }
                    if j < grid_size - 1 {
                        adj_list.push(map_ijkl_to_grid(i, j + 1, k, l, grid_size));
                    }
                    if k > 0 {
                        adj_list.push(map_ijkl_to_grid(i, j, k - 1, l, grid_size));
                    }
                    if k < grid_size - 1 {
                        adj_list.push(map_ijkl_to_grid(i, j, k + 1, l, grid_size));
                    }
                    if l > 0 {
                        adj_list.push(map_ijkl_to_grid(i, j, k, l - 1, grid_size));
                    }
                    if l < grid_size - 1 {
                        adj_list.push(map_ijkl_to_grid(i, j, k, l + 1, grid_size));
                    }
                    adj_lists.push(adj_list);
                }
            }
        }
    }
    adj_lists
}

pub fn generate_4d_grid_vectors_f32(grid_size: u32) -> Vec<Vec<f32>> {
    (0..grid_size)
        .flat_map(|i| {
            (0..grid_size).flat_map(move |j| {
                (0..grid_size).flat_map(move |k| {
                    (0..grid_size).map(move |l| vec![i as f32, j as f32, k as f32, l as f32])
                })
            })
        })
        .collect()
}

pub fn generate_4d_grid_vectors_i8(grid_size: i8) -> Vec<Vec<i8>> {
    (0..grid_size)
        .flat_map(|i| {
            (0..grid_size).flat_map(move |j| {
                (0..grid_size).flat_map(move |k| (0..grid_size).map(move |l| vec![i, j, k, l]))
            })
        })
        .collect()
}

pub fn generate_4d_grid_vectors_u8(grid_size: u8) -> Vec<Vec<u8>> {
    (0..grid_size)
        .flat_map(|i| {
            (0..grid_size).flat_map(move |j| {
                (0..grid_size).flat_map(move |k| (0..grid_size).map(move |l| vec![i, j, k, l]))
            })
        })
        .collect()
}

pub fn generate_circle_vectors(radius: f32, num_points: usize) -> Vec<Vec<f32>> {
    (0..num_points)
        .map(|i| {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / num_points as f32;
            vec![radius * theta.cos(), radius * theta.sin()]
        })
        .collect()
}

pub fn generate_circle_adj_list(num_points: u32) -> Vec<AdjacencyList<u32>> {
    (0..num_points)
        .map(|i| {
            AdjacencyList::from_iter_untrusted([
                (i + 1) % num_points,
                ((i + num_points) - 1) % num_points,
            ])
        })
        .collect()
}

pub fn generate_circle_with_various_radii_vectors(num_points: usize) -> Vec<Vec<f32>> {
    (0..num_points)
        .map(|i| {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / num_points as f32;
            let radius = 1.0 + 2.0 * (i % 7) as f32;
            vec![radius * theta.cos(), radius * theta.sin()]
        })
        .collect()
}
