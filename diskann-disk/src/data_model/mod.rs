/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod graph_layout_version;
pub use graph_layout_version::GraphLayoutVersion;

mod graph_metadata;
pub use graph_metadata::GraphMetadata;

mod graph_header;
pub use graph_header::GraphHeader;

mod cache;
pub use cache::{Cache, CachingStrategy};

pub mod graph_data_types;
pub use graph_data_types::{AdHoc, GraphDataType};

pub const FP_VECTOR_MEM_ALIGN: usize = 32;
