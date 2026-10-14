/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark Bloom candidate enumeration for private one-clause DNF pattern files.

use diskann_label_index::{EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType};
use serde_json::{json, Value};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
    time::Instant,
};

struct Case {
    id: String,
    query: EncodedLabelQuery<'static>,
    positives: u64,
    checksum: u64,
    nanos: Vec<u64>,
}

fn evaluate(query: &EncodedLabelQuery<'_>) -> (u64, u64) {
    let mut count = 0u64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    query.visit_matches(|id| {
        count += 1;
        checksum ^= u64::from(id);
        checksum = checksum.wrapping_mul(0x100_0000_01b3);
    });
    (count, checksum)
}

fn run(args: &[String]) -> Result<Value, Box<dyn Error>> {
    if args.len() < 5 {
        return Err(
            "usage: benchmark_bloom_patterns INDEX.bin REPS OUTPUT.json PATTERNS.txt [...]".into(),
        );
    }
    let repetitions: usize = args[2].parse()?;
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let output = Path::new(&args[3]);
    let partial = output.with_extension("json.part");
    if output.exists() || partial.exists() {
        return Err(format!(
            "benchmark output or partial file already exists: {}",
            output.display()
        )
        .into());
    }

    let index = EncodedLabelIndex::load(&args[1])?;
    let mut cases = Vec::new();
    for (file_index, path) in args[4..].iter().enumerate() {
        for (line_index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let clause = line?;
            if clause.is_empty() {
                return Err(format!("{path}:{}: empty pattern", line_index + 1).into());
            }
            let query = index.query(&[clause], FilterExpressionType::DNF)?;
            let (positives, checksum) = evaluate(&query);
            cases.push(Case {
                id: format!("F{:02}P{:02}", file_index + 1, line_index + 1),
                query,
                positives,
                checksum,
                nanos: Vec::with_capacity(repetitions),
            });
        }
    }
    if cases.is_empty() {
        return Err("pattern files contain no queries".into());
    }

    let case_count = cases.len();
    for repetition in 0..repetitions {
        for offset in 0..case_count {
            let case = &mut cases[(offset + repetition) % case_count];
            let start = Instant::now();
            let (positives, checksum) = evaluate(&case.query);
            let elapsed = u64::try_from(start.elapsed().as_nanos())?;
            if positives != case.positives || checksum != case.checksum {
                return Err(format!("{}: candidate IDs changed", case.id).into());
            }
            case.nanos.push(elapsed);
        }
    }

    let mut total_nanos = 0u64;
    let mut rows = Vec::with_capacity(cases.len());
    for case in cases {
        let mut nanos = case.nanos;
        nanos.sort_unstable();
        let sum: u64 = nanos.iter().sum();
        total_nanos += sum;
        rows.push(json!({
            "case_id": case.id,
            "bloom_positives": case.positives,
            "checksum": format!("{:016x}", case.checksum),
            "min_ms": nanos[0] as f64 / 1_000_000.0,
            "median_ms": nanos[nanos.len() / 2] as f64 / 1_000_000.0,
            "mean_ms": sum as f64 / repetitions as f64 / 1_000_000.0,
        }));
    }

    let result = json!({
        "index": args[1],
        "num_vectors": index.num_vectors(),
        "repetitions_per_case": repetitions,
        "case_count": rows.len(),
        "visits": rows.len() * repetitions,
        "mean_scan_ms": total_nanos as f64
            / (rows.len() * repetitions) as f64
            / 1_000_000.0,
        "cases": rows,
    });
    fs::write(&partial, serde_json::to_vec_pretty(&result)?)?;
    fs::rename(partial, output)?;
    Ok(result)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let result = run(&args)?;
    println!(
        "{} cases, {} visits, mean {} ms",
        result["case_count"], result["visits"], result["mean_scan_ms"]
    );
    Ok(())
}
