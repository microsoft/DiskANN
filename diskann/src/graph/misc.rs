/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// enum used to return the status of the vector that `consolidate_vector`
// was called on: Deleted if the vector was already deleted, and Complete
// if the vector was not deleted (and thus is now consolidated)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsolidateKind {
    /// Consolidate was called on a deleted vector.
    Deleted,

    /// Consolidate was called on valid vector, but retrieving the data for that vector
    /// failed with a transient error.
    FailedVectorRetrieval,

    /// Consolidate completed successfully.
    Complete,
}

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidate_enum() {
        // test the already deleted variant
        let delete_res_already_deleted = ConsolidateKind::Deleted;
        match delete_res_already_deleted {
            ConsolidateKind::Deleted => {}
            _ => panic!("Expected already deleted variant"),
        }

        // test the not deleted variant
        let delete_res_not_deleted = ConsolidateKind::Complete;
        match delete_res_not_deleted {
            ConsolidateKind::Complete => {}
            _ => panic!("Expected not deleted variant"),
        }
    }
}
