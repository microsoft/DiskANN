/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Semver-style version stamps embedded in every saved object.

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// A semver-style schema version attached to every saved record.
///
/// Each [`crate::save::Save`] / [`crate::load::Load`] impl declares its
/// `const VERSION: Version`. On load, the version recorded in the manifest is compared
/// against the declared version to dispatch between
/// [`Load::load`](crate::load::Load::load) and
/// [`Load::load_legacy`](crate::load::Load::load_legacy).
///
/// The framework treats versions as opaque pairs and only checks them for equality;
/// ordering / semver semantics are entirely up to the implementing type.
///
/// The `major.minor` format aids code readability: a reader can tell at a glance
/// that, for example, version `1.0` is compatible with version `1.2`.
///
/// On the wire, a `Version` is encoded as a single string of the form
/// `"major.minor"` (e.g. `"0.0"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

impl Version {
    /// Construct a [`Version`] from its two components.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl std::str::FromStr for Version {
    type Err = ParseVersionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('.');
        let major = parts.next().and_then(|s| s.parse::<u32>().ok());
        let minor = parts.next().and_then(|s| s.parse::<u32>().ok());
        match (major, minor, parts.next()) {
            (Some(major), Some(minor), None) => Ok(Version { major, minor }),
            _ => Err(ParseVersionError(s.to_owned())),
        }
    }
}

/// Error returned when a string cannot be parsed as a [`Version`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseVersionError(String);

impl std::fmt::Display for ParseVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown version {:?}: expected two `.`-separated u32 components",
            self.0,
        )
    }
}

impl std::error::Error for ParseVersionError {}

#[cfg(feature = "serde")]
impl Serialize for Version {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.collect_str(self)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct VersionVisitor;

        impl de::Visitor<'_> for VersionVisitor {
            type Value = Version;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a version string of the form \"major.minor\"")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Version, E> {
                v.parse().map_err(E::custom)
            }
        }

        de.deserialize_str(VersionVisitor)
    }
}

#[cfg(all(test, feature = "disk"))]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_dotted_string() {
        let json = serde_json::to_string(&Version::new(1, 2)).unwrap();
        assert_eq!(json, "\"1.2\"");
    }

    #[test]
    fn round_trips_through_json() {
        let v = Version::new(4, 5);
        let back: Version = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn rejects_malformed_strings() {
        for bad in ["\"1\"", "\"1.2.3\"", "\"1.x\"", "\"abc\""] {
            serde_json::from_str::<Version>(bad)
                .expect_err("malformed version string must be rejected");
        }
    }
}
