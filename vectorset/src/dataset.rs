/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_core::recall::Rows;
use diskann_utils::views::Matrix;
use serde::Deserialize;
use std::{
    fs::File,
    io::{Read, Seek},
    ops::Deref,
    path::{Path, PathBuf},
};
use thiserror::Error;

use crate::{DistanceMetric, Element, ElementType};

const MAX_DIM: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum DataType {
    #[serde(
        rename = "float32",
        alias = "f32",
        alias = "Float32",
        alias = "FLOAT32"
    )]
    F32,
    #[serde(
        rename = "int8",
        alias = "i8",
        alias = "Int8",
        alias = "INT8",
        alias = "I8"
    )]
    I8,
    #[serde(
        rename = "uint8",
        alias = "u8",
        alias = "Uint8",
        alias = "UINT8",
        alias = "U8"
    )]
    U8,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::I8 => std::mem::size_of::<i8>(),
            DataType::U8 => std::mem::size_of::<u8>(),
        }
    }
}

#[derive(Debug, Error)]
pub enum DatasetSpecError {
    #[error("spec has no vectors")]
    NoVectors,
    #[error("spec has no queries")]
    NoQueries,
    #[error("bad dimension of {0}: must be positive and less than {MAX_DIM}")]
    BadDim(usize),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad yaml: {0}")]
    BadYaml(#[from] serde_saphyr::Error),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct DatasetSpec {
    vectors: usize,
    queries: usize,
    dim: usize,
    data_type: DataType,
    metric: DistanceMetric,
    base_path: PathBuf,
    query_path: PathBuf,
    gt_path: PathBuf,
    step_gt_dir: PathBuf,
}

impl DatasetSpec {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, DatasetSpecError> {
        let f = File::open(path)?;
        let spec = serde_saphyr::from_reader(f)?;
        Self::validate(&spec)?;
        Ok(spec)
    }

    fn validate(&self) -> Result<(), DatasetSpecError> {
        if self.vectors == 0 {
            return Err(DatasetSpecError::NoVectors);
        }
        if self.queries == 0 {
            return Err(DatasetSpecError::NoQueries);
        }
        if self.dim == 0 || self.dim > MAX_DIM {
            return Err(DatasetSpecError::BadDim(self.dim));
        }

        Ok(())
    }
}

impl From<DataType> for ElementType {
    fn from(value: DataType) -> Self {
        match value {
            DataType::F32 => ElementType::F32,
            DataType::I8 => ElementType::I8,
            DataType::U8 => ElementType::U8,
        }
    }
}

#[derive(Debug, Error)]
pub enum RowBufError {
    #[error("error creating rowbuf")]
    Create,
}

#[derive(Debug)]
pub struct RowBuf<T: Element> {
    start: usize,
    buf: Matrix<T>,
}

impl<T: Element> RowBuf<T> {
    fn from_buffer(
        start: usize,
        dim: usize,
        count: usize,
        buffer: Vec<T>,
    ) -> Result<Self, RowBufError> {
        let buf = Matrix::try_from(buffer.into_boxed_slice(), count, dim)
            .map_err(|_| RowBufError::Create)?;
        Ok(Self { start, buf })
    }

    pub fn row(&self, index: usize) -> &[T] {
        assert!(index >= self.start && index - self.start < self.buf.nrows());

        self.buf.row(index - self.start)
    }

    pub fn start(&self) -> usize {
        self.start
    }
}

impl<T: Element> Deref for RowBuf<T> {
    type Target = Matrix<T>;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl<T: Element> Rows<T> for RowBuf<T> {
    fn nrows(&self) -> usize {
        self.buf.nrows()
    }

    fn ncols(&self) -> Option<usize> {
        Some(self.buf.ncols())
    }

    fn row(&self, i: usize) -> &[T] {
        RowBuf::row(self, i)
    }
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid dataset spec: {0}")]
    BadSpec(#[from] DatasetSpecError),
    #[error("missing path: {0}")]
    MissingPath(PathBuf),
    #[error("dimension specified ({1}) doesn't match dimension in {0} data file ({2})")]
    DimMismatch(String, usize, usize),
    #[error("{0} data file had wrong size")]
    WrongSize(String),
    #[error("{0} data file had different count ({1}) than expected ({2})")]
    WrongCount(String, usize, usize),
    #[error("found {0} ground truth entries, but expected {1}")]
    GtMismatch(usize, usize),
    #[error("gt dimension ({0}) must be positive and less than {MAX_DIM}")]
    BadGtDim(usize),
    #[error("index range ({0}..{1}) greater than max bound ({2})")]
    IndexRangeOutOfBounds(usize, usize, usize),
    #[error("rowbuf: {0}")]
    RowBuf(#[from] RowBufError),
    #[error("element type mismatch (expected: {0:?}; got {1:?})")]
    ElementType(ElementType, ElementType),
}

#[derive(Debug)]
pub struct Dataset {
    data_type: DataType,

    base_path: PathBuf,
    query_path: PathBuf,
    step_gt_dir: PathBuf,

    dim: usize,
    metric: DistanceMetric,
    vector_count: usize,
    query_count: usize,

    search_paths: Option<Vec<PathBuf>>,
}

impl Dataset {
    pub fn from_path<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        search_paths: Option<&[S]>,
    ) -> Result<Self, DatasetError> {
        let spec = DatasetSpec::from_path(&path)?;

        let base_path = Self::resolve_path(&spec.base_path, search_paths)?;
        let query_path = Self::resolve_path(&spec.query_path, search_paths)?;
        let gt_path = Self::resolve_path(&spec.gt_path, search_paths)?;
        let step_gt_dir = Self::resolve_path(&spec.step_gt_dir, search_paths)?;

        let dim = spec.dim;
        let metric = spec.metric;

        let (vector_count, base_dim, base_len) = Self::metadata(&base_path)?;
        if vector_count != spec.vectors {
            return Err(DatasetError::WrongCount(
                "base".to_string(),
                vector_count,
                spec.vectors,
            ));
        }
        if dim != base_dim {
            return Err(DatasetError::DimMismatch("base".to_string(), dim, base_dim));
        }
        let base_len_check = vector_count * base_dim * spec.data_type.size() + 8; // 8 bytes extra for the u32 count and dimension
        if base_len != base_len_check {
            return Err(DatasetError::WrongSize("base".to_string()));
        }

        let (query_count, query_dim, query_len) = Self::metadata(&query_path)?;
        if query_count != spec.queries {
            return Err(DatasetError::WrongCount(
                "query".to_string(),
                query_count,
                spec.queries,
            ));
        }
        if dim != query_dim {
            return Err(DatasetError::DimMismatch(
                "query".to_string(),
                dim,
                query_dim,
            ));
        }
        let query_len_check = query_count * query_dim * spec.data_type.size() + 8; // 8 bytes extra for the u32 count and dimension
        if query_len != query_len_check {
            return Err(DatasetError::WrongSize("query".to_string()));
        }

        let (gt_count, gt_dim, gt_len) = Self::metadata(&gt_path)?;
        if gt_dim == 0 || gt_dim > MAX_DIM {
            return Err(DatasetError::BadGtDim(gt_dim));
        }
        if gt_count != query_count {
            return Err(DatasetError::GtMismatch(gt_count, query_count));
        }
        let gt_len_check =
            gt_count * gt_dim * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()) + 8; // 8 bytes extra for the u32 count and dimension
        if gt_len != gt_len_check {
            return Err(DatasetError::WrongSize("gt".to_string()));
        }

        let search_paths = search_paths.map(|sp| {
            sp.iter()
                .map(|p| p.as_ref().to_owned())
                .collect::<Vec<PathBuf>>()
        });

        Ok(Dataset {
            data_type: spec.data_type,
            base_path,
            query_path,
            step_gt_dir,
            dim,
            metric,
            vector_count,
            query_count,
            search_paths,
        })
    }

    pub fn vector_count(&self) -> usize {
        self.vector_count
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn read_rows<T: Element>(
        &self,
        path: &Path,
        index: usize,
        count: usize,
        dim: usize,
    ) -> Result<RowBuf<T>, DatasetError> {
        let mut f = File::open(path)?;

        let offset = index * dim * std::mem::size_of::<T>() + 8;

        let mut row = vec![T::default(); dim * count];
        f.seek(std::io::SeekFrom::Start(offset as u64))?;
        f.read_exact(bytemuck::cast_slice_mut(&mut row))?;

        Ok(RowBuf::from_buffer(index, dim, count, row)?)
    }

    pub fn vectors<T: Element>(
        &self,
        index: usize,
        count: usize,
    ) -> Result<RowBuf<T>, DatasetError> {
        if T::ELEMENT_TYPE != self.data_type.clone().into() {
            return Err(DatasetError::ElementType(
                self.data_type.clone().into(),
                T::ELEMENT_TYPE,
            ));
        }

        if index >= self.vector_count || index + count > self.vector_count {
            return Err(DatasetError::IndexRangeOutOfBounds(
                index,
                count,
                self.vector_count,
            ));
        }

        self.read_rows::<T>(&self.base_path, index, count, self.dim)
    }

    pub fn queries<T: Element>(
        &self,
        index: usize,
        count: usize,
    ) -> Result<RowBuf<T>, DatasetError> {
        if T::ELEMENT_TYPE != self.data_type.clone().into() {
            return Err(DatasetError::ElementType(
                self.data_type.clone().into(),
                T::ELEMENT_TYPE,
            ));
        }

        if index >= self.query_count || index + count > self.query_count {
            return Err(DatasetError::IndexRangeOutOfBounds(
                index,
                count,
                self.query_count,
            ));
        }

        self.read_rows::<T>(&self.query_path, index, count, self.dim)
    }

    fn resolve_path<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        search_paths: Option<&[S]>,
    ) -> Result<PathBuf, DatasetError> {
        if path.as_ref().exists() {
            return Ok(path.as_ref().to_path_buf());
        }

        if !path.as_ref().is_relative() {
            return Err(DatasetError::MissingPath(path.as_ref().to_path_buf()));
        }

        if let Some(search_paths) = search_paths {
            for search_path in search_paths {
                let p = search_path.as_ref().join(&path);
                if p.exists() {
                    return Ok(p);
                }
            }
        }

        Err(DatasetError::MissingPath(path.as_ref().to_path_buf()))
    }

    fn metadata<P: AsRef<Path>>(path: P) -> Result<(usize, usize, usize), DatasetError> {
        let mut f = File::open(&path)?;

        let len = f.metadata()?.len() as usize;

        let mut count = 0u32;
        let mut dim = 0u32;

        f.read_exact(bytemuck::bytes_of_mut(&mut count))?;
        f.read_exact(bytemuck::bytes_of_mut(&mut dim))?;

        Ok((count as usize, dim as usize, len))
    }

    pub fn step_gt(
        &self,
        runbook_name: &str,
        step: usize,
    ) -> Result<(RowBuf<u32>, RowBuf<f32>), DatasetError> {
        let step_gt_path = PathBuf::new()
            .join(&self.step_gt_dir)
            .join(runbook_name)
            .join(format!("step{step}.gt100"));

        let resolved = Self::resolve_path(step_gt_path, self.search_paths.as_deref())?;

        // Step ground truth files carry their own dimension, which need not match `gt_path`'s.
        let (gt_count, gt_dim, _) = Self::metadata(&resolved)?;
        if gt_dim == 0 || gt_dim > MAX_DIM {
            return Err(DatasetError::BadGtDim(gt_dim));
        }

        let nrows = self.read_rows::<u32>(&resolved, 0, self.query_count, gt_dim)?;
        let drows = self.read_rows::<f32>(&resolved, gt_count, self.query_count, gt_dim)?;

        Ok((nrows, drows))
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    use crate::{
        dataset::{Dataset, DatasetError, DatasetSpec, DatasetSpecError},
        test_utils::create_test_yaml,
    };

    #[test]
    fn really_invalid_yaml() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            this is not valid
            "#,
        );

        let res = DatasetSpec::from_path(&path);
        assert_matches!(res, Err(DatasetSpecError::BadYaml(_)));
    }

    #[test]
    fn zero_dim() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 10
            queries: 10
            dim: 0
            data-type: float32
            metric: l2
            base-path: base
            query-path: query
            gt-path: gt
            step-gt-dir: .
            "#,
        );

        let res = DatasetSpec::from_path(&path);
        assert_matches!(res, Err(DatasetSpecError::BadDim(0)));
    }
    #[test]
    fn dim_too_big() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 10
            queries: 10
            dim: 120039102938123
            data-type: float32
            metric: l2
            base-path: base
            query-path: query
            gt-path: gt
            step-gt-dir: .
            "#,
        );

        let res = DatasetSpec::from_path(&path);
        assert_matches!(res, Err(DatasetSpecError::BadDim(120039102938123)));
    }

    #[test]
    fn spec_vector_count_zero() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 0
            queries: 10
            dim: 10
            data-type: "float32"
            metric: l2
            base-path: f32_zero_count.bin
            query-path: f32_good.bin
            gt-path: u32_good.bin
            step-gt-dir: .
            "#,
        );

        let res = Dataset::from_path(&path, Some(&["test_data"]));
        assert_matches!(res, Err(DatasetError::BadSpec(DatasetSpecError::NoVectors)));
    }

    #[test]
    fn vector_count_zero() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 10
            queries: 10
            dim: 10
            data-type: "float32"
            metric: l2
            base-path: f32_zero_count.bin
            query-path: f32_good.bin
            gt-path: u32_good.bin
            step-gt-dir: .
            "#,
        );

        let res = Dataset::from_path(&path, Some(&["test_data"]));
        assert_matches!(res, Err(DatasetError::WrongCount(_, 0, 10)));
    }

    #[test]
    fn spec_query_count_zero() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 10
            queries: 0
            dim: 10
            data-type: "float32"
            metric: l2
            base-path: f32_good.bin
            query-path: f32_zero_count.bin
            gt-path: u32_good.bin
            step-gt-dir: .
            "#,
        );

        let res = Dataset::from_path(&path, Some(&["test_data"]));
        assert_matches!(res, Err(DatasetError::BadSpec(DatasetSpecError::NoQueries)));
    }

    #[test]
    fn query_count_zero() {
        let (_tmpdir, path) = create_test_yaml(
            r#"
            vectors: 10
            queries: 10
            dim: 10
            data-type: "float32"
            metric: l2
            base-path: f32_good.bin
            query-path: f32_zero_count.bin
            gt-path: u32_good.bin
            step-gt-dir: .
            "#,
        );

        let res = Dataset::from_path(&path, Some(&["test_data"]));
        assert_matches!(res, Err(DatasetError::WrongCount(_, 0, 10)));
    }
}
