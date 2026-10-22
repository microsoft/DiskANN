/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(super) mod dto {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct Full;

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct KnnOptions;
}
