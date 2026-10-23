/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::{self, Display, Formatter},
    io::Write,
    sync::Arc,
};

use diskann_benchmark_runner::{
    self as dbr,
    files::InputFile,
    utils::{
        datatype::{AsDataType, DataType},
        fmt::KeyValue,
    },
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

use crate::utils::{datafiles::check_data_type_load, SimilarityMeasure};

pub(super) trait FromRaw: Sized {
    type Raw;
    fn from_raw(raw: &Self::Raw) -> anyhow::Result<Self>;
}

pub(super) trait FromRawWith: Sized {
    type Raw;
    fn from_raw_with(raw: &Self::Raw, checker: Option<&mut dbr::Checker>) -> anyhow::Result<Self>;
}

pub(super) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct Data {
        pub(super) data_type: DataType,
        pub(super) data: InputFile,
        pub(super) distance: SimilarityMeasure,
    }

    impl Data {
        pub(in crate::index::inmem2) fn example() -> Self {
            Self {
                data_type: DataType::Float32,
                data: InputFile::new("path/to/data"),
                distance: SimilarityMeasure::SquaredL2,
            }
        }
    }
}

/// A handle to a data-set living somewhere on disk.
///
/// Datasets are characterized by their data type, their source file, and their distance
/// metric.
#[derive(Debug)]
pub(super) struct Data {
    data_type: DataType,
    data: InputFile,
    distance: Metric,
}

impl Data {
    /// Create a new [`Data`].
    pub(super) fn new(data_type: DataType, data: InputFile, distance: Metric) -> Self {
        Self {
            data_type,
            data,
            distance,
        }
    }

    /// Return the [`Metric`] used for the dataset.
    pub(super) fn distance(&self) -> Metric {
        self.distance
    }

    /// Return the [`DataType`] for the dataset.
    pub(super) fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Load the dataset into memory.
    ///
    /// Returns an error if `T` does not match [`Self::data_type`], or if the load fails
    /// for any other reason (missing file, corrupt file etc).
    pub(super) fn load<T>(
        &self,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<rowmajor::Owned<T>>
    where
        T: AsDataType + bytemuck::Pod,
    {
        check_data_type_load::<T>(self.data_type)?;

        let data =
            crate::utils::datafiles::load_dataset(crate::utils::datafiles::BinFile(&self.data))?;

        if let Some(mut output) = output {
            writeln!(
                output,
                "Loaded {} points of dim {} as type {} from {}\n",
                data.nrows(),
                data.ncols(),
                T::DATA_TYPE,
                self.data.display(),
            )?;
        }

        Ok(data)
    }

    /// Load the dataset and packaged it into an [`Arc`].
    ///
    /// See: [`Self::load`].
    pub(super) fn load_arc<T>(
        &self,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Arc<rowmajor::Owned<T>>>
    where
        T: AsDataType + bytemuck::Pod,
    {
        self.load(output).map(Arc::new)
    }
}

impl FromRawWith for Data {
    type Raw = dto::Data;
    fn from_raw_with(raw: &dto::Data, checker: Option<&mut dbr::Checker>) -> anyhow::Result<Self> {
        let dto::Data {
            data_type,
            data,
            distance,
        } = raw;

        let mut data = data.clone();
        if let Some(checker) = checker {
            data.resolve(checker)?;
        }

        Ok(Self::new(*data_type, data, (*distance).into()))
    }
}

impl Display for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("data_type", &self.data_type);
        kv.push("data", &self.data);
        kv.push("distance", &self.distance);
        write!(f, "{}", kv)
    }
}
