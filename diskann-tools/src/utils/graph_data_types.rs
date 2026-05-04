/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_disk::data_model::AdHoc;
use diskann_vector::Half;

pub type GraphDataF32Vector = AdHoc<f32>;
pub type GraphDataHalfVector = AdHoc<Half>;
pub type GraphDataInt8Vector = AdHoc<i8>;
pub type GraphDataU8Vector = AdHoc<u8>;
