/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Instruction Set Architecture (ISA) selection for the multi-vector factories.

/// Instruction Set Architecture (ISA) selector for which multi-vector MaxSim or
/// TopK kernel to build.
///
/// `#[non_exhaustive]` so adding a variant (e.g. for a new in-tree kernel) is
/// not a breaking change. Deliberately **not** `Serialize`/`Deserialize` —
/// callers wanting JSON support maintain their own shadow enum and convert
/// via `From` / `TryFrom`, so the library is not pinned to a particular
/// serialization format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(non_camel_case_types)]
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
    /// Include unavailable ISAs so tests can also exercise rejection paths.
    #[cfg(test)]
    pub(crate) fn all() -> [Self; 6] {
        [
            Self::Auto,
            Self::Scalar,
            Self::X86_64_V3,
            Self::X86_64_V4,
            Self::Neon,
            Self::Reference,
        ]
    }

    /// Whether a kernel for this ISA can be built on the current host.
    /// Feature-gated variants may return `false` even when compiled for
    /// the matching target architecture.
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

// Returns Result<R, NotSupported>, where R is the builder's output.
// Builder, input, and reference expressions are evaluated only in the selected branch.
macro_rules! dispatch_isa {
    ($isa:expr, $builder:expr, $input:expr, $reference:expr $(,)?) => {{
        use diskann_wide::{Architecture as _, arch};
        use $crate::multi_vector::distance::isa::{MaxSimIsa, NotSupported};

        let isa = $isa;
        match isa {
            MaxSimIsa::Auto => Ok::<_, NotSupported>(arch::dispatch1_no_features($builder, $input)),
            MaxSimIsa::Scalar => Ok(arch::Scalar::new().run1($builder, $input)),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V3 => arch::x86_64::V3::new_checked()
                .ok_or(NotSupported {
                    isa,
                    reason: "AVX2/FMA unavailable on this CPU",
                })
                .map(|arch| arch.run1($builder, $input)),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V4 => arch::x86_64::V4::new_checked()
                .ok_or(NotSupported {
                    isa,
                    reason: "AVX-512 unavailable on this CPU",
                })
                .map(|arch| arch.run1($builder, $input)),
            #[cfg(target_arch = "aarch64")]
            MaxSimIsa::Neon => arch::aarch64::Neon::new_checked()
                .ok_or(NotSupported {
                    isa,
                    reason: "Neon unavailable on this CPU",
                })
                .map(|arch| arch.run1($builder, $input)),
            MaxSimIsa::Reference => Ok($reference),
            #[cfg(not(target_arch = "x86_64"))]
            MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => Err(NotSupported {
                isa,
                reason: "x86_64 target only",
            }),
            #[cfg(not(target_arch = "aarch64"))]
            MaxSimIsa::Neon => Err(NotSupported {
                isa,
                reason: "aarch64 target only",
            }),
        }
    }};
}

pub(super) use dispatch_isa;

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

/// Returned by [`build_max_sim`](super::build_max_sim) or
/// [`build_top_k`](super::build_top_k) when the requested ISA cannot be produced
/// on the current host (e.g. x86_64 V4 requested on a non-AVX512 CPU, or Neon
/// requested on x86_64).
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

#[cfg(test)]
mod tests {
    use std::{any::type_name, cell::Cell};

    use diskann_wide::{Architecture, arch};

    use super::*;

    struct Identify;

    impl<A: Architecture> arch::Target1<A, (&'static str, String), String> for Identify {
        fn run(self, _: A, input: String) -> (&'static str, String) {
            (type_name::<A>(), input)
        }
    }

    fn track<T>(calls: &Cell<usize>, value: T) -> T {
        calls.set(calls.get() + 1);
        value
    }

    #[test]
    fn dispatch_selects_isa_without_evaluating_other_branches() {
        for isa in MaxSimIsa::all() {
            let calls = [const { Cell::new(0) }; 3];
            let input = String::from("query");
            let result = dispatch_isa!(
                isa,
                track(&calls[0], Identify),
                track(&calls[1], input),
                track(&calls[2], ("reference", input)),
            );
            if !isa.is_available() {
                assert_eq!(result.unwrap_err().isa, isa);
                assert_eq!(calls.map(|c| c.get()), [0, 0, 0]);
                continue;
            }
            let expected = match isa {
                MaxSimIsa::Auto => arch::dispatch1_no_features(Identify, String::new()).0,
                MaxSimIsa::Scalar => type_name::<arch::Scalar>(),
                #[cfg(target_arch = "x86_64")]
                MaxSimIsa::X86_64_V3 => type_name::<arch::x86_64::V3>(),
                #[cfg(target_arch = "x86_64")]
                MaxSimIsa::X86_64_V4 => type_name::<arch::x86_64::V4>(),
                #[cfg(target_arch = "aarch64")]
                MaxSimIsa::Neon => type_name::<arch::aarch64::Neon>(),
                MaxSimIsa::Reference => "reference",
                _ => unreachable!("unavailable ISAs were handled above"),
            };
            let (selected, input) = result.unwrap();
            assert_eq!(selected, expected);
            assert_eq!(input, "query");
            assert_eq!(
                calls.map(|c| c.get()),
                if isa == MaxSimIsa::Reference {
                    [0, 0, 1]
                } else {
                    [1, 1, 0]
                }
            );
        }
    }
}
