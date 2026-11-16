/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[path = "search.rs"]
mod search;

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("neighbor-queue-lookup", LookupBenchmark)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LookupInput {
    seed: u64,
    samples: NonZeroUsize,
    fixtures: NonZeroUsize,
    blocks_per_sample: NonZeroUsize,
    lengths: Vec<usize>,
}

impl std::fmt::Display for LookupInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Queue lower bound: seed {}", self.seed)
    }
}

impl Input for LookupInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "neighbor-queue-lookup"
    }

    fn from_raw(raw: Self, _: &mut Checker) -> anyhow::Result<Self> {
        ensure!(!raw.lengths.is_empty(), "lengths must not be empty");
        raw.fixtures
            .get()
            .checked_mul(raw.blocks_per_sample.get())
            .context("lookups per sample overflow usize")?;
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        Self {
            seed: 20261114,
            samples: NonZeroUsize::new(9).unwrap(),
            fixtures: NonZeroUsize::new(256).unwrap(),
            blocks_per_sample: NonZeroUsize::new(100).unwrap(),
            lengths: vec![20, 40, 100, 200, 500, 1000, 2000, 8192],
        }
    }
}

struct LookupBenchmark;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Kernel {
    Linear,
    Binary,
    Hybrid128,
    Hybrid256,
    Hybrid512,
    Hybrid1024,
    Hybrid512Prefix256,
}

const KERNELS: [Kernel; 7] = [
    Kernel::Linear,
    Kernel::Binary,
    Kernel::Hybrid128,
    Kernel::Hybrid256,
    Kernel::Hybrid512,
    Kernel::Hybrid1024,
    Kernel::Hybrid512Prefix256,
];

#[derive(Serialize)]
struct LookupResult {
    length: usize,
    position: &'static str,
    samples: Vec<LookupSample>,
}

#[derive(Serialize)]
struct LookupSample {
    round: usize,
    method: Kernel,
    ns_per_lookup: f64,
}

struct LookupFixture {
    distances: Vec<f32>,
    target: f32,
}

#[inline(never)]
fn measure_lookup(
    fixtures: &[LookupFixture],
    blocks: usize,
    lookup: impl Fn(&[f32], f32) -> usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..blocks {
        for fixture in fixtures {
            black_box(lookup(
                black_box(&fixture.distances),
                black_box(fixture.target),
            ));
        }
    }
    start.elapsed().as_nanos() as f64 / (fixtures.len() * blocks) as f64
}

impl Benchmark for LookupBenchmark {
    type Input = LookupInput;
    type Output = Vec<LookupResult>;

    fn try_match(&self, _: &LookupInput, context: &MatchContext) -> Score {
        context.success(1)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Private queue lower-bound kernels (SIMD linear on x86)")
    }

    fn run(
        &self,
        input: &LookupInput,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let mut rng = StdRng::seed_from_u64(input.seed);
        let mut results = Vec::new();
        for &length in &input.lengths {
            for position in ["head", "middle", "tail", "random", "equal", "reject"] {
                let fixtures: Vec<_> = (0..input.fixtures.get())
                    .map(|_| {
                        let offset = rng.random_range(-1.0f32..1.0);
                        let distances: Vec<_> =
                            (0..length).map(|i| offset + (i / 3) as f32 * 0.5).collect();
                        let target = if length == 0 {
                            offset
                        } else {
                            match position {
                                "head" => distances[0] - 0.1,
                                "middle" => distances[length / 2] + 0.1,
                                "tail" => distances[length - 1],
                                "random" => rng
                                    .random_range(distances[0] - 0.1..distances[length - 1] + 0.1),
                                "equal" => distances[rng.random_range(0..length)],
                                "reject" => distances[length - 1] + 0.1,
                                _ => unreachable!(),
                            }
                        };
                        LookupFixture { distances, target }
                    })
                    .collect();
                for f in &fixtures {
                    let expected = f
                        .distances
                        .iter()
                        .position(|d| *d >= f.target)
                        .unwrap_or(length);
                    ensure!(search::linear(&f.distances, f.target) == expected);
                    ensure!(search::binary(&f.distances, f.target) == expected);
                    ensure!(search::hybrid::<128, 16>(&f.distances, f.target) == expected);
                    ensure!(search::hybrid::<256, 16>(&f.distances, f.target) == expected);
                    ensure!(search::hybrid::<512, 16>(&f.distances, f.target) == expected);
                    ensure!(search::hybrid::<1024, 16>(&f.distances, f.target) == expected);
                    ensure!(search::hybrid::<512, 256>(&f.distances, f.target) == expected);
                }
                let mut samples = Vec::new();
                // One untimed warmup round; rotate kernels in recorded rounds.
                for round in 0..=input.samples.get() {
                    for offset in 0..KERNELS.len() {
                        let method = KERNELS[(round + offset) % KERNELS.len()];
                        let blocks = if round == 0 {
                            1
                        } else {
                            input.blocks_per_sample.get()
                        };
                        let ns_per_lookup = match method {
                            Kernel::Linear => measure_lookup(&fixtures, blocks, search::linear),
                            Kernel::Binary => measure_lookup(&fixtures, blocks, search::binary),
                            Kernel::Hybrid128 => {
                                measure_lookup(&fixtures, blocks, search::hybrid::<128, 16>)
                            }
                            Kernel::Hybrid256 => {
                                measure_lookup(&fixtures, blocks, search::hybrid::<256, 16>)
                            }
                            Kernel::Hybrid512 => {
                                measure_lookup(&fixtures, blocks, search::hybrid::<512, 16>)
                            }
                            Kernel::Hybrid1024 => {
                                measure_lookup(&fixtures, blocks, search::hybrid::<1024, 16>)
                            }
                            Kernel::Hybrid512Prefix256 => {
                                measure_lookup(&fixtures, blocks, search::hybrid::<512, 256>)
                            }
                        };
                        if round != 0 {
                            samples.push(LookupSample {
                                round: round - 1,
                                method,
                                ns_per_lookup,
                            });
                        }
                    }
                }
                writeln!(output, "lookup len={length} position={position}")?;
                results.push(LookupResult {
                    length,
                    position,
                    samples,
                });
            }
        }
        Ok(results)
    }
}
