/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use diskann_utils::views;

use crate::utils::Bridge;

// Compatibility with ANNError.
impl From<Bridge<diskann_quantization::views::ChunkOffsetError>> for ANNError {
    #[track_caller]
    fn from(value: Bridge<diskann_quantization::views::ChunkOffsetError>) -> Self {
        ANNError::log_pq_error(value.into_inner())
    }
}

// Compatibility with ANNError.
impl From<Bridge<diskann_quantization::views::ChunkViewError>> for ANNError {
    #[track_caller]
    fn from(value: Bridge<diskann_quantization::views::ChunkViewError>) -> Self {
        ANNError::log_pq_error(value.into_inner())
    }
}

// Compatibility with ANNError.
impl<T: views::DenseData> From<Bridge<views::TryFromError<T>>> for ANNError {
    #[track_caller]
    fn from(value: Bridge<views::TryFromError<T>>) -> Self {
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

    fn test_error<F, E>(f: F)
    where
        F: FnOnce() -> E,
        E: std::error::Error,
        ANNError: From<E>,
    {
        let err = f();
        let message = format!("{}", err);

        let ann = ANNError::from(err);
        assert_eq!(ann.kind(), ANNErrorKind::PQError);
        let formatted = ann.to_string();
        assert!(formatted.contains(&message));
    }

    #[test]
    fn test_chunk_offsets_error() {
        // Offsets not monotonic.
        let offsets = [0, 1, 1, 5];
        test_error(|| {
            diskann_quantization::views::ChunkOffsetsView::new(&offsets)
                .bridge_err()
                .unwrap_err()
        });
    }

    #[test]
    fn test_chunk_view_error() {
        let offsets = [0, 1, 2, 5];
        // Data is too long.
        let data = vec![0; offsets.last().unwrap() + 1];

        test_error(|| {
            let offsets = diskann_quantization::views::ChunkOffsetsView::new(&offsets).unwrap();
            diskann_quantization::views::ChunkView::new(data.as_slice(), offsets)
                .bridge_err()
                .unwrap_err()
        });
    }

    #[test]
    fn test_try_from() {
        let ncols = 5;
        let nrows = 3;
        let data = vec![0; ncols * nrows];

        test_error(|| {
            views::MatrixView::try_from(&*data, nrows, ncols + 1)
                .bridge_err()
                .unwrap_err()
        });
    }
}
