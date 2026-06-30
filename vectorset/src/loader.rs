/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::{Result, anyhow};
use std::{collections::HashMap, marker::PhantomData, mem, path::Path, sync::Arc};
use tokio::{fs::File, io::AsyncReadExt, sync::Mutex};

const BATCH_SIZE: usize = 1024;

/// Loads base or query vectors from a given path and allow iteration over them.
pub struct DatasetLoader<T: bytemuck::Pod> {
    file: Mutex<(File, usize)>,
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
        let mut f = self.file.lock().await;

        let mut count;
        let mut first_id;
        loop {
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

            if count == 0 {
                continue;
            }

            break;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Write a `.bin` dataset file (i32 num_vectors, i32 dim, then row-major data).
    fn write_bin<T: bytemuck::Pod>(
        dir: &TempDir,
        name: &str,
        dim: usize,
        rows: &[Vec<T>],
    ) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(rows.len() as i32).to_le_bytes()).unwrap();
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for row in rows {
            f.write_all(bytemuck::cast_slice(row)).unwrap();
        }
        path
    }

    /// Write raw bytes after a header (used for malformed-file tests).
    fn write_raw(dir: &TempDir, name: &str, num_vectors: i32, dim: i32, body: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&num_vectors.to_le_bytes()).unwrap();
        f.write_all(&dim.to_le_bytes()).unwrap();
        f.write_all(body).unwrap();
        path
    }

    #[tokio::test]
    async fn new_reads_header() {
        let dir = TempDir::new().unwrap();
        let rows = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let path = write_bin(&dir, "base.bin", 2, &rows);

        let loader = DatasetLoader::<f32>::new(&path).await.unwrap();
        assert_eq!(loader.len(), 3);
        assert_eq!(loader.dim(), 2);
        assert_eq!(loader.batch_size(), BATCH_SIZE);
    }

    #[tokio::test]
    async fn next_reads_all_vectors_in_order() {
        let dir = TempDir::new().unwrap();
        let rows = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let path = write_bin(&dir, "base.bin", 2, &rows);
        let loader = DatasetLoader::<f32>::new(&path).await.unwrap();

        let mut buf = Vec::new();
        let (count, first_id) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, 3);
        assert_eq!(first_id, 0);
        assert_eq!(&buf[..count * 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Subsequent call yields nothing and clears the buffer.
        let (count, first_id) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, 0);
        assert_eq!(first_id, 3);
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn next_spans_multiple_batches() {
        let dir = TempDir::new().unwrap();
        let total = BATCH_SIZE + 5;
        let rows: Vec<Vec<f32>> = (0..total).map(|i| vec![i as f32]).collect();
        let path = write_bin(&dir, "base.bin", 1, &rows);
        let loader = DatasetLoader::<f32>::new(&path).await.unwrap();

        let mut buf = Vec::new();
        let (count, first_id) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, BATCH_SIZE);
        assert_eq!(first_id, 0);
        assert_eq!(buf[0], 0.0);

        let (count, first_id) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, 5);
        assert_eq!(first_id, BATCH_SIZE);
        assert_eq!(buf[0], BATCH_SIZE as f32);

        let (count, _) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn next_on_empty_dataset() {
        let dir = TempDir::new().unwrap();
        let path = write_bin::<f32>(&dir, "empty.bin", 4, &[]);
        let loader = DatasetLoader::<f32>::new(&path).await.unwrap();

        let mut buf = Vec::new();
        let (count, first_id) = loader.next(&mut buf).await.unwrap();
        assert_eq!(count, 0);
        assert_eq!(first_id, 0);
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn next_errors_on_truncated_vector() {
        let dir = TempDir::new().unwrap();
        // Header claims 2 vectors of dim 3, but only 4 f32 values follow
        // (one full vector plus a partial second one).
        let body: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let path = write_raw(&dir, "bad.bin", 2, 3, &body);
        let loader = DatasetLoader::<f32>::new(&path).await.unwrap();

        let mut buf = Vec::new();
        assert!(loader.next(&mut buf).await.is_err());
    }

    #[tokio::test]
    async fn load_returns_all_vectors() {
        let dir = TempDir::new().unwrap();
        let rows = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let path = write_bin(&dir, "base.bin", 3, &rows);

        let loaded = DatasetLoader::<f32>::load(&path).await.unwrap();
        assert_eq!(*loaded, rows);
    }

    #[tokio::test]
    async fn load_groundtruth_builds_map() {
        let dir = TempDir::new().unwrap();
        let num_queries = 2i32;
        let num_neighbors = 2i32;
        let ids: [u32; 4] = [10, 11, 20, 21];
        let dists: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

        let mut body = Vec::new();
        body.extend(bytemuck::cast_slice::<u32, u8>(&ids));
        body.extend(bytemuck::cast_slice::<f32, u8>(&dists));
        let path = write_raw(&dir, "gt.bin", num_queries, num_neighbors, &body);

        let map = DatasetLoader::<f32>::load_groundtruth(&path).await.unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&0], vec![(10, 0.1), (11, 0.2)]);
        assert_eq!(map[&1], vec![(20, 0.3), (21, 0.4)]);
    }
}
