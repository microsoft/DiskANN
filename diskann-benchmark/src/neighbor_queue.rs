/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fixed-capacity queue updates, excluding input generation and queue reset.

use std::{hint::black_box, io::Write, num::NonZeroUsize, time::Instant};

use anyhow::{ensure, Context};
use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    Benchmark, Checker, Checkpoint, Input, Output, Registry,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

type Queue = NeighborPriorityQueue<u32>;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("neighbor-queue", QueueBenchmark)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueueInput {
    seed: u64,
    samples: NonZeroUsize,
    queues_per_sample: NonZeroUsize,
    blocks_per_sample: NonZeroUsize,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    capacity: NonZeroUsize,
    initial_size: usize,
    input_size: usize,
    initial_cutoff_survivors: usize,
    placement: Placement,
    order: Order,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Placement {
    Head,
    Uniform,
    Tail,
    Middle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Order {
    Shuffled,
    Ascending,
    Descending,
}

impl QueueInput {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(!self.cases.is_empty(), "cases must not be empty");
        for (i, c) in self.cases.iter().enumerate() {
            ensure!(
                c.initial_size <= c.capacity.get(),
                "case {i}: initial_size must not exceed capacity"
            );
            ensure!(
                c.initial_cutoff_survivors <= c.input_size,
                "case {i}: initial_cutoff_survivors must not exceed input_size"
            );
            ensure!(
                c.initial_size == c.capacity.get() || c.initial_cutoff_survivors == c.input_size,
                "case {i}: a partially full queue has no initial cutoff"
            );
            let ids = c.initial_size.checked_add(c.input_size);
            ensure!(
                ids.is_some_and(|n| u32::try_from(n).is_ok()),
                "case {i}: unique IDs must fit u32"
            );
        }
        self.queues_per_sample
            .get()
            .checked_mul(self.blocks_per_sample.get())
            .context("updates per sample overflow usize")?;
        Ok(())
    }
}

impl std::fmt::Display for QueueInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neighbor queue: {} cases, seed {}",
            self.cases.len(),
            self.seed
        )
    }
}

impl Input for QueueInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "neighbor-queue"
    }

    fn from_raw(raw: Self, _: &mut Checker) -> anyhow::Result<Self> {
        raw.validate()?;
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        Self {
            seed: 20261113,
            samples: NonZeroUsize::new(9).unwrap(),
            queues_per_sample: NonZeroUsize::new(8).unwrap(),
            blocks_per_sample: NonZeroUsize::new(2).unwrap(),
            cases: vec![Case {
                capacity: NonZeroUsize::new(20).unwrap(),
                initial_size: 20,
                input_size: 8,
                initial_cutoff_survivors: 8,
                placement: Placement::Uniform,
                order: Order::Shuffled,
            }],
        }
    }
}

struct QueueBenchmark;

impl Benchmark for QueueBenchmark {
    type Input = QueueInput;
    type Output = Vec<CaseResult>;

    fn try_match(&self, _: &QueueInput, context: &MatchContext) -> Score {
        context.success(1)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fixed-capacity u32 queue: insert vs batch_insert")
    }

    fn run(
        &self,
        input: &QueueInput,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Vec<CaseResult>> {
        let mut rng = StdRng::seed_from_u64(input.seed);
        let mut results = Vec::with_capacity(input.cases.len());
        for (i, case) in input.cases.iter().enumerate() {
            let result = run_case(input, case, i, &mut rng)
                .with_context(|| format!("neighbor-queue case {i}: {case:?}"))?;
            write!(
                output,
                "L={} initial={} N={} initial_cutoff_survivors={} {:?}/{:?}: median ns/update",
                case.capacity,
                case.initial_size,
                case.input_size,
                case.initial_cutoff_survivors,
                case.placement,
                case.order
            )?;
            for method in METHODS {
                let values = result
                    .samples
                    .iter()
                    .flat_map(|s| &s.timings)
                    .filter(|t| t.method == method)
                    .map(|t| t.ns_per_update)
                    .collect();
                write!(output, " {method:?}={:.2}", median(values))?;
            }
            writeln!(output)?;
            results.push(result);
        }
        Ok(results)
    }
}

#[derive(Debug, Serialize)]
struct CaseResult {
    case: Case,
    mean_initial_cutoff_survivors: f64,
    mean_scalar_admissions: f64,
    mean_scalar_moved_entries: f64,
    mean_final_new_entries: f64,
    samples: Vec<Sample>,
}

#[derive(Debug, Serialize)]
struct Sample {
    round: usize,
    first_method: Method,
    timings: Vec<Timing>,
}

#[derive(Debug, Serialize)]
struct Timing {
    method: Method,
    ns_per_update: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Method {
    Insert,
    BatchInsert,
}

const METHODS: [Method; 2] = [Method::Insert, Method::BatchInsert];

impl Method {
    fn update(self, queue: &mut Queue, buffer: &mut Vec<Neighbor<u32>>) {
        match self {
            Self::Insert => insert(queue, buffer),
            Self::BatchInsert => queue.batch_insert(buffer),
        }
    }
}

fn insert(queue: &mut Queue, buffer: &mut Vec<Neighbor<u32>>) {
    for neighbor in buffer.drain(..) {
        queue.insert(neighbor);
    }
}

struct Fixture {
    base: Queue,
    queue: Queue,
    input: Vec<Neighbor<u32>>,
    buffer: Vec<Neighbor<u32>>,
}

impl Fixture {
    fn reset(&mut self) {
        self.queue.clear();
        for neighbor in self.base.iter() {
            self.queue.insert(neighbor);
        }
    }
}

fn fixture(case: &Case, rng: &mut StdRng) -> Fixture {
    let capacity = case.capacity.get();
    let offset = rng.random_range(0.0f32..0.25);
    let mut base = Queue::new(capacity);
    let neighbor = |i| Neighbor::new(i as u32, offset + 1.0 + (i + 1) as f32 / capacity as f32);
    for i in 0..case.initial_size {
        base.insert(neighbor(i));
    }

    let best = base.iter().next().map_or(offset + 1.0, |n| *n.distance());
    let worst = base.iter().last().map_or(offset + 2.0, |n| *n.distance());
    let middle = best + (worst - best) * 0.5;
    let mut input: Vec<_> = (0..case.input_size)
        .map(|i| {
            let distance = if i >= case.initial_cutoff_survivors {
                worst + rng.random_range(0.1f32..1.0)
            } else {
                match case.placement {
                    Placement::Head => best - rng.random_range(0.1f32..1.0),
                    Placement::Uniform => best + rng.random::<f32>() * (worst - best),
                    Placement::Tail => worst,
                    Placement::Middle => middle,
                }
            };
            Neighbor::new((case.initial_size + i) as u32, distance)
        })
        .collect();
    match case.order {
        Order::Shuffled => input.shuffle(rng),
        Order::Ascending => input.sort_by(diskann::neighbor::ord::fast_distance),
        Order::Descending => input.sort_by(diskann::neighbor::ord::reverse(
            diskann::neighbor::ord::fast_distance,
        )),
    }
    Fixture {
        queue: Queue::new(capacity),
        buffer: Vec::with_capacity(case.input_size),
        base,
        input,
    }
}

fn contents(queue: &Queue) -> Vec<(u32, u32)> {
    queue
        .iter()
        .map(|n| (*n.id(), n.distance().to_bits()))
        .collect()
}

fn unvisited(queue: &mut Queue) -> Vec<(u32, u32)> {
    std::iter::from_fn(|| queue.closest_notvisited())
        .map(|n| (*n.id(), n.distance().to_bits()))
        .collect()
}

// Counts come from an untimed scalar model, not instrumentation of either hot path.
fn verify(f: &Fixture) -> anyhow::Result<[usize; 4]> {
    let candidates = &f.input;
    let mut reference = f.base.clone();
    let mut input = candidates.to_vec();
    insert(&mut reference, &mut input);
    let expected = contents(&reference);
    let expected_unvisited = unvisited(&mut reference);
    for method in METHODS {
        let mut queue = f.base.clone();
        let mut buffer = candidates.to_vec();
        let (capacity, ptr) = (buffer.capacity(), buffer.as_ptr());
        method.update(&mut queue, &mut buffer);
        ensure!(
            buffer.is_empty() && buffer.capacity() == capacity && buffer.as_ptr() == ptr,
            "{method:?}: buffer was not cleared with allocation preserved"
        );
        ensure!(
            contents(&queue) == expected,
            "{method:?}: ordered queue mismatch"
        );
        ensure!(
            unvisited(&mut queue) == expected_unvisited,
            "{method:?}: unvisited sequence mismatch"
        );
    }
    let capacity = f.base.capacity();
    let mut distances: Vec<_> = f.base.iter().map(|n| *n.distance()).collect();
    let worst = (distances.len() == capacity).then(|| distances[capacity - 1]);
    let survivors = candidates
        .iter()
        .filter(|n| !n.distance().is_nan() && worst.is_none_or(|d| *n.distance() <= d))
        .count();
    let (mut admissions, mut moves) = (0, 0);
    for n in candidates {
        let d = *n.distance();
        if d.is_nan() || (distances.len() == capacity && d > distances[capacity - 1]) {
            continue;
        }
        let index = distances.partition_point(|x| *x < d);
        if distances.len() == capacity {
            distances.pop();
        }
        moves += distances.len() - index;
        distances.insert(index, d);
        admissions += 1;
    }
    ensure!(
        distances
            .iter()
            .map(|d| d.to_bits())
            .eq(expected.iter().map(|(_, d)| *d)),
        "scalar work model disagrees with production queue"
    );
    let new_entries = expected
        .iter()
        .filter(|(id, _)| candidates.iter().any(|n| n.id() == id))
        .count();
    Ok([survivors, admissions, moves, new_entries])
}

#[inline(never)]
fn measure(
    fixtures: &mut [Fixture],
    blocks: usize,
    mut update: impl FnMut(&mut Queue, &mut Vec<Neighbor<u32>>),
) -> f64 {
    let mut elapsed = 0u128;
    for _ in 0..blocks {
        for f in fixtures.iter_mut() {
            f.reset();
            f.buffer.clear();
            f.buffer.extend_from_slice(&f.input);
        }
        let start = Instant::now();
        for f in fixtures.iter_mut() {
            update(black_box(&mut f.queue), black_box(&mut f.buffer));
            black_box(&f.queue);
            black_box(&f.buffer);
        }
        elapsed += start.elapsed().as_nanos();
    }
    elapsed as f64 / (fixtures.len() * blocks) as f64
}

fn run_case(
    input: &QueueInput,
    case: &Case,
    case_index: usize,
    rng: &mut StdRng,
) -> anyhow::Result<CaseResult> {
    let mut fixtures: Vec<_> = (0..input.queues_per_sample.get())
        .map(|_| fixture(case, rng))
        .collect();
    let mut work = [0usize; 4];
    for f in &fixtures {
        for (total, count) in work.iter_mut().zip(verify(f)?) {
            *total += count;
        }
    }
    // Warm up all paths, including allocation, outside recorded samples.
    for method in METHODS {
        measure(&mut fixtures, 1, |q, b| method.update(q, b));
    }
    let mut samples = Vec::with_capacity(input.samples.get());
    for round in 0..input.samples.get() {
        let first = (round + case_index) % METHODS.len();
        let mut timings = Vec::with_capacity(METHODS.len());
        for offset in 0..METHODS.len() {
            let method = METHODS[(first + offset) % METHODS.len()];
            let blocks = input.blocks_per_sample.get();
            // Dispatch outside the timed loop so each method can inline its update.
            let ns_per_update = match method {
                Method::Insert => measure(&mut fixtures, blocks, insert),
                Method::BatchInsert => measure(&mut fixtures, blocks, |q, b| q.batch_insert(b)),
            };
            timings.push(Timing {
                method,
                ns_per_update,
            });
        }
        samples.push(Sample {
            round,
            first_method: METHODS[first],
            timings,
        });
    }
    let means = work.map(|v| v as f64 / fixtures.len() as f64);
    Ok(CaseResult {
        case: case.clone(),
        mean_initial_cutoff_survivors: means[0],
        mean_scalar_admissions: means[1],
        mean_scalar_moved_entries: means[2],
        mean_final_new_entries: means[3],
        samples,
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_cases_preserve_queue_and_buffer_contracts() {
        let mut rng = StdRng::seed_from_u64(42);
        for capacity in [1, 16, 127] {
            for input_size in [0, 1, 8, 31] {
                for placement in [
                    Placement::Head,
                    Placement::Uniform,
                    Placement::Tail,
                    Placement::Middle,
                ] {
                    for order in [Order::Shuffled, Order::Ascending, Order::Descending] {
                        for initial_size in [0, capacity / 2, capacity] {
                            let case = Case {
                                capacity: NonZeroUsize::new(capacity).unwrap(),
                                initial_size,
                                input_size,
                                initial_cutoff_survivors: input_size,
                                placement,
                                order,
                            };
                            verify(&fixture(&case, &mut rng)).unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut input = QueueInput::example();
        input.cases[0].initial_cutoff_survivors = 9;
        assert!(input.validate().is_err());
        input.cases[0].initial_cutoff_survivors = 8;
        input.cases[0].initial_size = 21;
        assert!(input.validate().is_err());
        input.cases.clear();
        assert!(input.validate().is_err());
    }

    #[test]
    fn candidate_cutoff_and_visited_cursor_are_checked() {
        let mut rng = StdRng::seed_from_u64(72);
        let case = Case {
            capacity: NonZeroUsize::new(20).unwrap(),
            initial_size: 20,
            input_size: 8,
            initial_cutoff_survivors: 4,
            placement: Placement::Tail,
            order: Order::Shuffled,
        };
        let mut f = fixture(&case, &mut rng);
        for _ in 0..10 {
            f.base.closest_notvisited();
        }
        assert_eq!(
            f.input
                .iter()
                .filter(|n| *n.distance() == *f.base.get(19).distance())
                .count(),
            4
        );
        assert_eq!(verify(&f).unwrap()[0], 4);
    }

    #[test]
    fn case_schema_excludes_sequence_and_search_parameters() {
        let mut value = serde_json::to_value(QueueInput::example()).unwrap();
        value["cases"][0]["batch_size"] = serde_json::json!(8);
        assert!(serde_json::from_value::<QueueInput>(value).is_err());
        let mut value = serde_json::to_value(QueueInput::example()).unwrap();
        value["methods"] = serde_json::json!(["insert"]);
        assert!(serde_json::from_value::<QueueInput>(value).is_err());
        let mut value = serde_json::to_value(QueueInput::example()).unwrap();
        value["cases"][0]["visited_prefix"] = serde_json::json!(10);
        assert!(serde_json::from_value::<QueueInput>(value).is_err());
        let mut value = serde_json::to_value(QueueInput::example()).unwrap();
        value["cases"][0]["eligible_count"] = serde_json::json!(8);
        assert!(serde_json::from_value::<QueueInput>(value).is_err());
    }

    #[test]
    fn paired_samples_record_insert_and_batch_insert_only() {
        let mut input = QueueInput::example();
        input.samples = NonZeroUsize::new(4).unwrap();
        input.queues_per_sample = NonZeroUsize::new(2).unwrap();
        input.blocks_per_sample = NonZeroUsize::new(1).unwrap();
        input.validate().unwrap();
        let mut rng = StdRng::seed_from_u64(input.seed);
        let result = run_case(&input, &input.cases[0], 0, &mut rng).unwrap();
        assert_eq!(result.samples.len(), 4);
        for sample in result.samples {
            assert_eq!(sample.timings.len(), 2);
            assert!(sample.timings.iter().any(|t| t.method == Method::Insert));
            assert!(sample
                .timings
                .iter()
                .any(|t| t.method == Method::BatchInsert));
        }
    }

    #[test]
    fn cutoff_work_counts_and_measurement_output() {
        let mut input = QueueInput::example();
        input.samples = NonZeroUsize::new(3).unwrap();
        input.queues_per_sample = NonZeroUsize::new(2).unwrap();
        input.blocks_per_sample = NonZeroUsize::new(1).unwrap();
        input.cases[0].initial_cutoff_survivors = 0;
        let mut rng = StdRng::seed_from_u64(input.seed);
        let result = run_case(&input, &input.cases[0], 0, &mut rng).unwrap();
        assert_eq!(result.mean_initial_cutoff_survivors, 0.0);
        assert_eq!(result.mean_scalar_admissions, 0.0);
        assert_eq!(result.mean_scalar_moved_entries, 0.0);
        assert_eq!(result.mean_final_new_entries, 0.0);
        assert_eq!(result.samples.len(), 3);
        for sample in result.samples {
            assert_eq!(sample.timings.len(), 2);
            assert!(sample.timings.iter().all(|t| t.ns_per_update > 0.0));
        }
    }
}
