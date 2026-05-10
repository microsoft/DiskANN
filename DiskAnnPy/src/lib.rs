/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

cfg_if::cfg_if! {
    // ISSUE-2024/11/04-mchisholm. I had some trouble linking the DiskAnnPy
    // module on aarch64 (arm64). Skip it for now.
    if #[cfg(target_arch = "x86_64")] {
        use pyo3::prelude::*;
        pub mod utils;
        use crate::utils::{
            BatchSearchResultWithStats, DataType, MetricPy, SearchResult,
        };
        mod async_memory_index;
        mod build_async_memory_index;
        mod build_disk_index;
        mod static_disk_index;
        mod bftree_index;
        mod build_bftree_index;
        mod quantization;

        use crate::utils::ANNErrorPy;
        use crate::quantization::{
            MinMaxPreprocessedQuery, MinMaxQuantizer, ProductPreprocessedQuery, ProductQuantizer,
            QuantizerBase,
        };
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{filter::LevelFilter, fmt, EnvFilter};

        fn init_subscriber() {
            let fmt_layer = fmt::layer().with_target(true);

            let filter_layer = EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy();

            tracing_subscriber::registry()
                .with(filter_layer)
                .with(fmt_layer)
                .init();
        }

        /// A Python module implemented in Rust.
        #[pymodule]
        fn _diskannpy(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
            //this enables the logger to print when desired
            init_subscriber();

            m.add_function(wrap_pyfunction!(build_disk_index::build_disk_index, m)?)?;
            m.add_function(wrap_pyfunction!(
                build_async_memory_index::build_memory_index,
                m
            )?)?;
            m.add_class::<static_disk_index::StaticDiskIndexF32>()?;
            m.add_class::<static_disk_index::StaticDiskIndexU8>()?;
            m.add_class::<static_disk_index::StaticDiskIndexInt8>()?;
            m.add_class::<async_memory_index::AsyncMemoryIndexF32>()?;
            m.add_class::<async_memory_index::AsyncMemoryIndexU8>()?;
            m.add_class::<async_memory_index::AsyncMemoryIndexInt8>()?;
            m.add_class::<bftree_index::BfTreeIndexF32>()?;
            m.add_class::<bftree_index::BfTreeIndexU8>()?;
            m.add_class::<bftree_index::BfTreeIndexInt8>()?;

            m.add_class::<DataType>()?;
            m.add_class::<MetricPy>()?;
            m.add_class::<SearchResult>()?;
            m.add_class::<BatchSearchResultWithStats>()?;
            m.add_class::<ANNErrorPy>()?;
            m.add_class::<QuantizerBase>()?;
            m.add_class::<MinMaxQuantizer>()?;
            m.add_class::<MinMaxPreprocessedQuery>()?;
            m.add_class::<ProductQuantizer>()?;
            m.add_class::<ProductPreprocessedQuery>()?;
            Ok(())
        }
    }
}
