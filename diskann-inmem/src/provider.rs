/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};

use crate::{
    store::Primary,
    layer::QueryDistance,
};

#[derive(Debug)]
pub struct Provider<T> {
    primary: Primary,
    layer: T,
}

////////////
// Search //
////////////

pub struct SearchAccessor<'a> {
    provider: &'a Provider,
}

