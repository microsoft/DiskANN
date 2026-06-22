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
/// The framework treats versions as opaque triples and only checks them for equality;
/// ordering / semver semantics are entirely up to the implementing type.
///
/// On the wire, a `Version` is encoded as a single string of the form
/// `"major.minor.patch"` (e.g. `"0.0.0"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    /// Construct a [`Version`] from its three components.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for Version {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.collect_str(&format_args!(
            "{}.{}.{}",
            self.major, self.minor, self.patch
        ))
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct VersionVisitor;

        impl de::Visitor<'_> for VersionVisitor {
            type Value = Version;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a version string of the form \"major.minor.patch\"")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Version, E> {
                let mut parts = v.split('.');
                let major = parts.next().and_then(|s| s.parse::<u32>().ok());
                let minor = parts.next().and_then(|s| s.parse::<u32>().ok());
                let patch = parts.next().and_then(|s| s.parse::<u32>().ok());
                match (major, minor, patch, parts.next()) {
                    (Some(major), Some(minor), Some(patch), None) => Ok(Version {
                        major,
                        minor,
                        patch,
                    }),
                    _ => Err(E::custom(format!(
                        "unknown version {:?}: expected three `.`-separated u32 components",
                        v,
                    ))),
                }
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
        let json = serde_json::to_string(&Version::new(1, 2, 3)).unwrap();
        assert_eq!(json, "\"1.2.3\"");
    }

    #[test]
    fn round_trips_through_json() {
        let v = Version::new(4, 5, 6);
        let back: Version = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn rejects_malformed_strings() {
        for bad in ["\"1.2\"", "\"1.2.3.4\"", "\"1.x.3\"", "\"abc\""] {
            serde_json::from_str::<Version>(bad)
                .expect_err("malformed version string must be rejected");
        }
    }
}
