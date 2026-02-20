use anyhow::{Result, anyhow};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use std::{
    collections::HashMap,
    io::SeekFrom,
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    sync::Mutex,
};

/// Loads base or query vectors from a given path and allow iteration over them.
#[allow(dead_code)]
pub struct DatasetLoader<T: bytemuck::Pod> {
    file: Mutex<(File, Vec<PathBuf>, usize)>,
    paths: Vec<PathBuf>,
    num_vectors: usize,
    dim: usize,
    batch_size: usize,
    headerless: bool,
    type_: PhantomData<T>,
}

impl<T: bytemuck::Pod> DatasetLoader<T> {
    pub async fn new_with_headerless_dim<P: AsRef<Path> + Clone>(
        paths: &[P],
        headerless_with_dim: Option<usize>,
    ) -> Result<Arc<Self>> {
        let mut paths: Vec<PathBuf> = paths
            .iter()
            .rev()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        let headerless = headerless_with_dim.is_some();

        // Calculate total vectors in all paths
        let mut num_vectors = 0usize;
        let mut first_file = true;
        let mut dim = 0usize;
        for p in paths.iter() {
            let mut file = File::open(p).await?;

            if !headerless {
                num_vectors += file.read_i32_le().await? as usize;

                if first_file {
                    dim = file.read_i32_le().await? as usize;
                } else {
                    let d = file.read_i32_le().await? as usize;
                    if d != dim {
                        return Err(anyhow!(
                            "path {} dimension mismatch: expected {} but got {}",
                            p.display(),
                            dim,
                            d
                        ));
                    }
                }
            } else {
                dim = headerless_with_dim.unwrap();
                num_vectors += file.metadata().await?.len() as usize / dim;
            }

            first_file = false;
        }

        let orig_paths = paths.clone();
        let file = if let Some(p) = paths.pop() {
            let mut file = File::open(p).await?;

            if !headerless {
                // Skip header which was already parsed.
                let _ = file.read_i32_le().await? as usize;
                let _ = file.read_i32_le().await? as usize;
            }

            file
        } else {
            return Err(anyhow!("expected at least one base path"));
        };

        let file = Mutex::new((file, paths, 0));
        Ok(Arc::new(Self {
            file,
            paths: orig_paths,
            num_vectors,
            dim,
            batch_size: 1024,
            headerless,
            type_: PhantomData,
        }))
    }

    pub async fn new<P: AsRef<Path> + Clone>(paths: &[P]) -> Result<Arc<Self>> {
        Self::new_with_headerless_dim(paths, None).await
    }

    async fn advance_file(&self) -> Result<bool> {
        let mut f = self.file.lock().await;

        let mut file = if let Some(p) = f.1.pop() {
            File::open(p).await?
        } else {
            return Ok(true);
        };

        if !self.headerless {
            // Skip the header which is already read
            let _ = file.read_i32_le().await? as usize;
            let _ = file.read_i32_le().await? as usize;
        }

        f.0 = file;

        Ok(false)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Reset dataset to first vector.
    #[allow(dead_code)]
    pub async fn reset(&self) -> Result<()> {
        let mut paths = self.paths.clone();
        let file = if let Some(p) = paths.pop() {
            let mut file = File::open(p).await?;

            if !self.headerless {
                // Skip header which was already parsed.
                let _ = file.read_i32_le().await? as usize;
                let _ = file.read_i32_le().await? as usize;
            }

            file
        } else {
            return Err(anyhow!("expected at least one base path"));
        };

        let mut f = self.file.lock().await;
        *f = (file, paths, 0);

        Ok(())
    }

    /// Seek to a specific vector.
    ///
    /// If you call `seek_to(N)` then the next `next(...)` call will load vector N.
    #[allow(dead_code)]
    pub async fn seek_to(&self, index: usize) -> Result<()> {
        self.reset().await?;

        let mut f = self.file.lock().await;
        let mut pos = 0;
        loop {
            let count = self.file_count(&mut f.0).await?;
            if pos + count > index {
                break;
            }
            pos += count;

            // advance to next file
            f.0 = if let Some(p) = f.1.pop() {
                File::open(p).await?
            } else {
                return Err(anyhow!("ran out of files while seeking"));
            };
            f.2 = pos;
        }

        let skip_count = index - pos;
        let header_len = if self.headerless { 0u64 } else { 8u64 };
        let _ =
            f.0.seek(SeekFrom::Start(
                header_len + (skip_count * mem::size_of::<T>() * self.dim) as u64,
            ))
            .await?;

        Ok(())
    }

    #[allow(dead_code)]
    async fn file_count(&self, f: &mut File) -> Result<usize> {
        let size = f.metadata().await?.len() as usize;
        let count = if self.headerless {
            size / mem::size_of::<T>() / self.dim
        } else {
            (size - 8) / mem::size_of::<T>() / self.dim
        };
        Ok(count)
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

                first_id = f.2;
                if f.2 >= self.num_vectors {
                    buffer.clear();
                    return Ok((0, first_id));
                }

                buffer.resize(self.batch_size * self.dim, T::zeroed());

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

                count = self.batch_size - elements_left / self.dim;
            }

            if count == 0 {
                if self.advance_file().await? {
                    return Ok((0, first_id));
                }

                continue;
            }

            break;
        }

        let mut f = self.file.lock().await;
        f.2 += count;

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

#[allow(dead_code)]
pub async fn sample_latin_hypercube<T: bytemuck::Pod>(
    dataset: Arc<DatasetLoader<T>>,
    samples: usize,
) -> Result<Vec<Vec<T>>> {
    let total_vectors = dataset.len();
    let dim = dataset.dim();

    let mut rng = StdRng::seed_from_u64(0xaf2f5fa0b5161acf);
    let mut result = vec![vec![T::zeroed(); dim]; samples];
    let mut v = vec![T::zeroed(); dim];

    // sample a random partitions down the diagonal
    for (s, res) in result.iter_mut().enumerate() {
        for (idx, val) in res.iter_mut().enumerate() {
            let step = total_vectors / samples;
            let row = rng.random_range(s * step..(s + 1) * step);

            dataset.seek_to(row).await?;
            let _ = dataset.next(&mut v).await?;
            let value = v[idx];
            *val = value;
        }
    }

    // shuffle the dimensions between the vectors for random sampling
    for start_idx in 0..samples {
        for dim_idx in 0..dim {
            let swap_idx = rng.random_range(0..samples);
            let swap = result[start_idx][dim_idx];
            result[start_idx][dim_idx] = result[swap_idx][dim_idx];
            result[swap_idx][dim_idx] = swap;
        }
    }

    Ok(result)
}
