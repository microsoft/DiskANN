/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use diskann_utils::{strided};

use crate::utils::Bridge;

impl From<Bridge<strided::TryFromError>> for ANNError {
    #[track_caller]
    fn from(value: Bridge<strided::TryFromError>) -> Self {
        ANNError::new(value.into_inner())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::BridgeErr;

    #[test]
    fn test_conversion() {
        let nrows = 5;
        let ncols = 3;

        let x = vec![u8::default(); nrows * ncols];

        // Provided the incorrect dimensions.
        let err = strided::Ref::try_from_data(&x, nrows, ncols + 1, ncols + 1)
            .bridge_err()
            .unwrap_err();
        let message = format!("{}", err);

        let ann = ANNError::from(err);
        let formatted = ann.to_string();
        assert!(formatted.contains(&message));
    }
}
