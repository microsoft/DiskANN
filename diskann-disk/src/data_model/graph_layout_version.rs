/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{cmp::Ordering, fmt, io::Cursor};

use byteorder::{LittleEndian, ReadBytesExt};
use diskann::{ANNError, ANNResult};

/// Graph layout version. In the format of `major.minor`.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GraphLayoutVersion {
    pub major: u32,
    pub minor: u32,
}

impl GraphLayoutVersion {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    pub fn major_version(&self) -> u32 {
        self.major
    }

    pub fn minor_version(&self) -> u32 {
        self.minor
    }

    /// Serialize the `GraphLayoutVersion` object to a byte vector with 8 bytes.
    /// Layout:
    /// | MajorVersion (4 bytes) | MinorVersion (4 bytes) |
    /// The layout_version contains two parts. The first 32 bits are the major version number in u32 format, and the last 32 bits are the minor version number in u32 format.
    /// Backward incompatible layout changes should increment the major version number while backward compatible changes increments the minor version number.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(8);
        buffer.extend_from_slice(self.major.to_le_bytes().as_ref());
        buffer.extend_from_slice(self.minor.to_le_bytes().as_ref());
        buffer
    }
}

impl PartialOrd for GraphLayoutVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphLayoutVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => self.minor.cmp(&other.minor),
            major_ordering => major_ordering,
        }
    }
}

impl fmt::Display for GraphLayoutVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

impl<'a> TryFrom<&'a [u8]> for GraphLayoutVersion {
    type Error = ANNError;
    /// Try creating a new `GraphLayoutVersion` object from a byte slice. The try_from syntax is used here instead of from because this operation can fail.
    /// Layout:
    /// | MajorVersion (4 bytes) | MinorVersion (4 bytes) |
    fn try_from(value: &'a [u8]) -> ANNResult<Self> {
        if value.len() < std::mem::size_of::<GraphLayoutVersion>() {
            Err(ANNError::log_parse_slice_error(
                "&[u8]".to_string(),
                "GraphLayoutVersion".to_string(),
                "The given bytes are not long enough to create a valid graph layout version."
                    .to_string(),
            ))
        } else {
            let mut cursor = Cursor::new(&value);
            let major = cursor.read_u32::<LittleEndian>()?;
            let minor = cursor.read_u32::<LittleEndian>()?;

            Ok(Self::new(major, minor))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::GraphLayoutVersion;

    #[test]
    fn test_graph_layout_version_creation() {
        let version = GraphLayoutVersion::new(1, 0);
        assert_eq!(version.major_version(), 1);
        assert_eq!(version.minor_version(), 0);
    }

    #[test]
    fn test_graph_layout_version_comparison() {
        let v1 = GraphLayoutVersion::new(1, 0);
        let v2 = GraphLayoutVersion::new(1, 1);
        let v3 = GraphLayoutVersion::new(2, 0);
        let v4 = GraphLayoutVersion::new(2, 1);
        let v5 = GraphLayoutVersion::new(2, 1);

        assert_eq!(v1.partial_cmp(&v2), Some(Ordering::Less));
        assert_eq!(v2.partial_cmp(&v3), Some(Ordering::Less));
        assert_eq!(v3.partial_cmp(&v4), Some(Ordering::Less));
        assert_eq!(v4.partial_cmp(&v5), Some(Ordering::Equal));
    }

    #[test]
    fn test_graph_layout_version_ordering() {
        let v1 = GraphLayoutVersion::new(1, 0);
        let v2 = GraphLayoutVersion::new(1, 1);
        let v3 = GraphLayoutVersion::new(2, 0);
        let v4 = GraphLayoutVersion::new(2, 1);

        assert_eq!(v1.cmp(&v2), Ordering::Less);
        assert_eq!(v2.cmp(&v3), Ordering::Less);
        assert_eq!(v3.cmp(&v4), Ordering::Less);
        assert_eq!(v4.cmp(&v4), Ordering::Equal);
    }

    #[test]
    fn test_graph_layout_version() {
        // happy case
        let version_bytes = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let version = GraphLayoutVersion::try_from(&version_bytes[..]).unwrap();
        assert_eq!(version.major_version(), 1);
        assert_eq!(version.minor_version(), 2);

        let bytes = vec![3; std::mem::size_of::<GraphLayoutVersion>() - 1]; // bytes are too short to create a valid graph layout version
        let result = GraphLayoutVersion::try_from(&bytes[..]);
        assert!(result.is_err());
    }
}
