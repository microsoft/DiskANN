// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! ISA selector for the multi-vector MaxSim factory.

/// Which ISA to use when building a MaxSim kernel.
///
/// Not `Serialize`/`Deserialize` by design — callers maintain their own
/// shadow enum for their serialization format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum MaxSimIsa {
    /// Pick the highest ISA the host CPU supports.
    Auto,
    /// Pure-scalar (emulated SIMD) kernel — always available.
    Scalar,
    /// x86_64 AVX2 + FMA.
    X86_64_V3,
    /// x86_64 AVX-512.
    X86_64_V4,
    /// AArch64 Neon.
    Neon,
    /// Non-SIMD reference fallback. Slow; serves as a correctness baseline.
    Reference,
}

impl std::fmt::Display for MaxSimIsa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::X86_64_V3 => "x86-64-v3",
            Self::X86_64_V4 => "x86-64-v4",
            Self::Neon => "neon",
            Self::Reference => "reference",
        };
        f.write_str(s)
    }
}

/// The requested ISA is not available on this host.
#[derive(Debug, Clone, Copy)]
pub struct NotSupported {
    pub isa: MaxSimIsa,
    pub reason: &'static str,
}

impl std::fmt::Display for NotSupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} not supported: {}", self.isa, self.reason)
    }
}

impl std::error::Error for NotSupported {}
