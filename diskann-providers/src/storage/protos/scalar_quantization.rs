/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

pub use super::generated::scalar_quantization::{ScalarQuantizer, Version};

impl ScalarQuantizer {
    const CURRENT_PROTO_LAYOUT_VERSION: Version = Version {
        major: 0,
        minor: 1,
        patch: 0,
    };

    pub fn from(
        quantizer: &diskann_quantization::scalar::ScalarQuantizer,
        compressed_data_file_name: String,
    ) -> Self {
        Self {
            version: Some(Self::CURRENT_PROTO_LAYOUT_VERSION),
            scale: quantizer.scale(),
            shift: quantizer.shift().to_vec(),
            mean_norm: quantizer.mean_norm(),
            compressed_data_file_name,
        }
    }
}

impl TryFrom<ScalarQuantizer> for diskann_quantization::scalar::ScalarQuantizer {
    type Error = ProtoConversionError;

    fn try_from(proto: ScalarQuantizer) -> Result<Self, Self::Error> {
        let version = proto
            .version
            .ok_or(ProtoConversionError::MissingProtoLayoutVersion)?;

        if version != ScalarQuantizer::CURRENT_PROTO_LAYOUT_VERSION {
            return Err(ProtoConversionError::UnsupportedProtoLayoutVersion {
                expected: vec![ScalarQuantizer::CURRENT_PROTO_LAYOUT_VERSION],
                got: version,
            });
        }

        Ok(Self::new(proto.scale, proto.shift, proto.mean_norm))
    }
}

#[derive(Debug, Error)]
pub enum ProtoConversionError {
    #[error("Proto Layout version is missing")]
    MissingProtoLayoutVersion,

    #[error(
        "Proto Layout version mismatch: expected one of {:?}, but got {:?}",
        expected,
        got
    )]
    UnsupportedProtoLayoutVersion {
        expected: Vec<Version>,
        got: Version,
    },
}

#[cfg(test)]
mod tests {
    use super::{ProtoConversionError, ScalarQuantizer, Version};

    #[test]
    fn test_default_version() {
        let version = Version::default();
        assert_eq!(version.major, 0);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn test_default_scalar_quantizer() {
        let quantizer = ScalarQuantizer::default();
        assert_eq!(quantizer.version, None);
        assert_eq!(quantizer.scale, 0.0);
        assert_eq!(quantizer.shift.len(), 0);
        assert_eq!(quantizer.mean_norm, None);
        assert_eq!(quantizer.compressed_data_file_name, "");
    }

    #[test]
    fn test_from() {
        let scale = 3.64;
        let shift = vec![1.0, 2.0, 3.0, 4.0];
        let mean_norm = Some(2.71);

        let quant =
            diskann_quantization::scalar::ScalarQuantizer::new(scale, shift.clone(), mean_norm);
        let file_name = "my_data.bin".to_string();
        let proto = ScalarQuantizer::from(&quant, file_name.clone());

        assert_eq!(
            proto.version,
            Some(ScalarQuantizer::CURRENT_PROTO_LAYOUT_VERSION)
        );
        assert_eq!(proto.scale, scale);
        assert_eq!(proto.shift, shift);
        assert_eq!(proto.mean_norm, mean_norm);
        assert_eq!(proto.compressed_data_file_name, file_name);
    }

    #[test]
    fn test_try_from_proto_success() {
        let scale = 1.23;
        let shift = vec![0.5, 0.6, 0.7];
        let mean_norm = Some(9.87);
        let proto = ScalarQuantizer {
            version: Some(ScalarQuantizer::CURRENT_PROTO_LAYOUT_VERSION),
            scale,
            shift: shift.clone(),
            mean_norm,
            compressed_data_file_name: "whatever".to_string(),
        };

        let quant = diskann_quantization::scalar::ScalarQuantizer::try_from(proto).unwrap();

        assert_eq!(quant.scale(), scale);
        assert_eq!(quant.shift(), shift.as_slice());
        assert_eq!(quant.mean_norm(), mean_norm);
    }

    #[test]
    fn test_try_from_proto_version_missing() {
        let proto = ScalarQuantizer {
            version: None,
            scale: 1.67,
            shift: vec![0.5, 0.6, 0.7],
            mean_norm: None,
            compressed_data_file_name: String::new(),
        };

        let result = diskann_quantization::scalar::ScalarQuantizer::try_from(proto);

        match result {
            Err(ProtoConversionError::MissingProtoLayoutVersion) => {}
            _ => panic!("expected MissingProtoLayoutVersion error variant"),
        }
    }

    #[test]
    fn test_try_from_proto_version_mismatch() {
        let wrong_version = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let proto = ScalarQuantizer {
            version: Some(wrong_version),
            scale: 1.67,
            shift: vec![0.5, 0.6, 0.7],
            mean_norm: None,
            compressed_data_file_name: String::new(),
        };

        let result = diskann_quantization::scalar::ScalarQuantizer::try_from(proto);

        match result {
            Err(ProtoConversionError::UnsupportedProtoLayoutVersion { expected, got }) => {
                assert_eq!(
                    expected,
                    vec![ScalarQuantizer::CURRENT_PROTO_LAYOUT_VERSION]
                );
                assert_eq!(got, wrong_version);
            }
            _ => panic!("expected UnsupportedProtoLayoutVersion error variant"),
        }
    }
}
