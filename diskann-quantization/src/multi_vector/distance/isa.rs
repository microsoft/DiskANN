// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Instruction Set Architecture (ISA) selector for the multi-vector MaxSim
//! factory.

/// Instruction Set Architecture (ISA) selector for which multi-vector MaxSim
/// kernel to build.
///
/// `#[non_exhaustive]` so adding a variant (e.g. for a new in-tree kernel) is
/// not a breaking change. Deliberately **not** `Serialize`/`Deserialize` —
/// callers wanting JSON support maintain their own shadow enum and convert
/// via `From` / `TryFrom`, so the library is not pinned to a particular
/// serialization format.
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

impl MaxSimIsa {
    /// Whether a kernel for this ISA can be built on the current host.
    /// Variants that depend on CPU features (`X86_64_V3`, `X86_64_V4`,
    /// `Neon`) may return `false` even when the crate is compiled for the
    /// matching target architecture. `Auto`, `Scalar`, and `Reference` are
    /// always available.
    pub fn is_available(self) -> bool {
        match self {
            Self::Auto | Self::Scalar | Self::Reference => true,
            #[cfg(target_arch = "x86_64")]
            Self::X86_64_V3 => diskann_wide::arch::x86_64::V3::new_checked().is_some(),
            #[cfg(target_arch = "x86_64")]
            Self::X86_64_V4 => diskann_wide::arch::x86_64::V4::new_checked().is_some(),
            #[cfg(not(target_arch = "x86_64"))]
            Self::X86_64_V3 | Self::X86_64_V4 => false,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => diskann_wide::arch::aarch64::Neon::new_checked().is_some(),
            #[cfg(not(target_arch = "aarch64"))]
            Self::Neon => false,
        }
    }
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

/// Returned by [`build_max_sim`](super::build_max_sim) when the requested
/// ISA cannot be produced on the current host (e.g. x86_64 V4 requested on
/// a non-AVX512 CPU, or Neon requested on x86_64).
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
