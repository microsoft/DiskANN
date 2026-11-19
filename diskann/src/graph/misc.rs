/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// enum used to encode the algorithmic choices for inplace delete
// the first term indicates what is used to approximate the in-neighbors
// the second term indicates what is used to approximate the replace
// candidates
// also includes any params specific to that choice
#[derive(Copy, Clone, Debug)]
pub enum InplaceDeleteMethod {
    VisitedAndTopK { k_value: usize, l_value: usize },
    TwoHopAndOneHop,
    OneHop,
}
