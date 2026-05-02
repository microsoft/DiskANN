/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::{Result, anyhow};
use std::{
    collections::HashMap,
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs::File, io::AsyncReadExt, sync::Mutex};

const BATCH_SIZE: usize = 1024;

/// Loads base or query vectors from a given path and allow iteration over them.
#[allow(dead_code)]
pub struct DatasetLoader<T: bytemuck::Pod> {
    file: Mutex<(File, usize)>,
    path: PathBuf,
    num_vectors: usize,
    dim: usize,
    type_: PhantomData<T>,
}

impl<T: bytemuck::Pod> DatasetLoader<T> {
    pub async fn new<P: AsRef<Path> + Clone>(path: P) -> Result<Arc<Self>> {
        let path = path.as_ref().to_path_buf();

        // Calculate total vectors in all paths
        let mut file = File::open(&path).await?;
        let num_vectors = file.read_i32_le().await? as usize;
        let dim = file.read_i32_le().await? as usize;

        Ok(Arc::new(Self {
            file: Mutex::new((file, 0)),
            path,
            num_vectors,
            dim,
            type_: PhantomData,
        }))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn batch_size(&self) -> usize {
        BATCH_SIZE
    }

    /// Load the next vectors into `buffer`.
    ///
    /// Returns (count, first_id) where `count` is the number of vectors loaded
    /// and `first_id` is the id of the first vector.
    pub async fn next(&self, buffer: &mut Vec<T>) -> Result<(usize, usize)> {
        let mut count;
        let mut first_id;
        loop {
            {
                let mut f = self.file.lock().await;

                first_id = f.1;
                if f.1 >= self.num_vectors {
                    buffer.clear();
                    return Ok((0, first_id));
                }

                buffer.resize(BATCH_SIZE * self.dim, T::zeroed());

                let mut buf: &mut [u8] = bytemuck::cast_slice_mut::<T, u8>(&mut *buffer);
                while let bytes_read = f.0.read(buf).await?
                    && bytes_read > 0
                {
                    buf = &mut buf[bytes_read..];
                }

                let elements_left = buf.len() / mem::size_of::<T>();
                if !buf.is_empty() && !elements_left.is_multiple_of(self.dim) {
                    return Err(anyhow!("unexpected EOF"));
                }

                count = BATCH_SIZE - elements_left / self.dim;
            }

            if count == 0 {
                continue;
            }

            break;
        }

        let mut f = self.file.lock().await;
        f.1 += count;

        Ok((count, first_id))
    }

    /// Load the entire dataset into a Vec.
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<Vec<Vec<T>>>> {
        let mut file = File::open(path).await?;
        let num_vectors = file.read_i32_le().await? as usize;
        let dim = file.read_i32_le().await? as usize;

        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut v = vec![T::zeroed(); dim];
            file.read_exact(bytemuck::cast_slice_mut(&mut v)).await?;
            vectors.push(v);
        }
        Ok(Arc::new(vectors))
    }

    pub async fn load_groundtruth<P: AsRef<Path>>(
        path: P,
    ) -> Result<Arc<HashMap<u32, Vec<(u32, f32)>>>> {
        let mut file = File::open(&path).await?;

        let num_queries = file.read_i32_le().await? as usize;
        let num_neighbors = file.read_i32_le().await? as usize;

        let mut nbuf = vec![0u32; num_queries * num_neighbors];
        let mut dbuf = vec![0f32; num_queries * num_neighbors];
        file.read_exact(bytemuck::cast_slice_mut(&mut nbuf)).await?;
        file.read_exact(bytemuck::cast_slice_mut(&mut dbuf)).await?;

        let id_dists: Vec<(u32, f32)> = nbuf.iter().copied().zip(dbuf.iter().copied()).collect();

        let mut map = HashMap::with_capacity(num_queries);

        for i in 0..num_queries {
            let start = i * num_neighbors;
            let end = start + num_neighbors;
            map.insert(i as u32, id_dists[start..end].to_vec());
        }

        Ok(Arc::new(map))
    }
}
