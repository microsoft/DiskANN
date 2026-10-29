/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod generator;
pub(crate) use generator::validate_data_generation_input;
pub use generator::QuantDataGenerator;

pub(crate) mod pq;
pub use pq::pq_generation::{PQGeneration, PQGenerationContext};
pub use pq::PQData;

mod compressor;
pub use compressor::QuantCompressor;

mod search_data;
pub(crate) use search_data::SearchDataInner;
pub use search_data::{SearchData, SearchQuantization};

mod spherical;
pub use spherical::generate_spherical_q3;
