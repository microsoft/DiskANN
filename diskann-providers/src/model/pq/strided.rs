/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use diskann_utils::{strided, views};

use crate::utils::Bridge;

// Compatibility with ANNError.
impl<T: views::DenseData> From<Bridge<strided::TryFromError<T>>> for ANNError {
    #[track_caller]
    fn from(value: Bridge<strided::TryFromError<T>>) -> Self {
        ANNError::log_pq_error(value.into_inner())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann::ANNErrorKind;

    use super::*;
    use crate::utils::BridgeErr;

    #[test]
    fn test_conversion() {
        let nrows = 5;
        let ncols = 3;

        let x = vec![u8::default(); nrows * ncols];

        // Provided the incorrect dimensions.
        let err = strided::StridedView::try_from(&x, nrows, ncols + 1, ncols + 1)
            .bridge_err()
            .unwrap_err();
        let message = format!("{}", err);

        let ann = ANNError::from(err);
        assert_eq!(ann.kind(), ANNErrorKind::PQError);
        let formatted = ann.to_string();
        assert!(formatted.contains(&message));
    }
}
