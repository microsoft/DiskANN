/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Microbenchmark isolating the per-node `is_match` cost of live filter matching.
//!
//! It compares multiple representations of the *same* corpus of per-document attribute
//! sets, evaluating the *same* predicate over the *same* probed node ids:
//!
//! * `treemap`  — the current production path: [`InlineAttributeIndex`]
//!   (`HashMap<u32, RoaringTreemap>`) frozen into a live provider whose `is_match`
//!   walks the encoded predicate calling `RoaringTreemap::contains` per terminal.
//! * `csr`      — a flat CSR layout (`offsets: Vec<u32>` + sorted `values: Vec<u16>`):
//!   a match is a contiguous-slice membership test on the node's own row.
//! * `posting`  — one `RoaringBitmap` per attribute id (the doc-ids carrying it):
//!   a match is an `AND`/`OR` of `posting[term].contains(node)`.
//!
//! Before timing, all representations are asserted to agree on every probe, so the benchmark
//! measures only *how* the match decision is computed, not *what* it decides.
//!
//! Tunables (env): `LF_BENCH_N` corpus size (default 1_000_000),
//! `LF_BENCH_PROBES` ids probed per iteration (default 50_000).

use criterion::{criterion_group, Criterion, Throughput};
use diskann::graph::index::QueryLabelProvider;
use diskann_label_filter::attribute::Attribute;
use diskann_label_filter::{
    ASTExpr, CompareOp, InlineAttributeIndex, InlineAttributeIndexBitslice,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;
use serde_json::json;
use std::hint::black_box;
use std::sync::Arc;

/// Number of distinct attribute ids (mirrors the report's 548 geo + 48 market = 596).
const VOCAB: u16 = 596;

/// A predicate over integer attribute-id terminals, shared by all three eval paths so
/// they perform logically identical work.
#[derive(Clone)]
enum Pred {
    Term(u16),
    And(Vec<Pred>),
    Or(Vec<Pred>),
}

impl Pred {
    /// CSR eval: membership tests against the node's sorted attribute slice.
    fn eval_csr(&self, node_attrs: &[u16]) -> bool {
        match self {
            Pred::Term(id) => node_attrs.binary_search(id).is_ok(),
            Pred::And(v) => v.iter().all(|p| p.eval_csr(node_attrs)),
            Pred::Or(v) => v.iter().any(|p| p.eval_csr(node_attrs)),
        }
    }

    /// Posting eval: membership tests against per-attribute doc-id bitmaps.
    fn eval_posting(&self, node: u32, posting: &[RoaringBitmap]) -> bool {
        match self {
            Pred::Term(id) => posting[*id as usize].contains(node),
            Pred::And(v) => v.iter().all(|p| p.eval_posting(node, posting)),
            Pred::Or(v) => v.iter().any(|p| p.eval_posting(node, posting)),
        }
    }

    /// Convert to the string-keyed [`ASTExpr`] consumed by the production live provider.
    fn to_ast(&self) -> ASTExpr {
        match self {
            Pred::Term(id) => ASTExpr::Compare {
                field: format!("L{id}"),
                op: CompareOp::Eq(json!(true)),
            },
            Pred::And(v) => ASTExpr::And(v.iter().map(Pred::to_ast).collect()),
            Pred::Or(v) => ASTExpr::Or(v.iter().map(Pred::to_ast).collect()),
        }
    }
}

/// Skewed label sampler: small ids are much more common (Zipf-like), so `AND`
/// predicates over low ids retain meaningful overlap instead of collapsing to empty.
fn sample_label(rng: &mut StdRng) -> u16 {
    let u: f64 = rng.random();
    let id = (f64::from(VOCAB) * u.powf(2.5)).floor() as u16;
    id.min(VOCAB - 1)
}

/// Build the shared corpus: each doc gets a small sorted, deduped set of attribute ids.
fn build_corpus(n: usize, rng: &mut StdRng) -> Vec<Vec<u16>> {
    let mut corpus = Vec::with_capacity(n);
    for _ in 0..n {
        let k = 1 + rng.random_range(0..4u32); // 1..=4 draws before dedup
        let mut attrs: Vec<u16> = (0..k).map(|_| sample_label(rng)).collect();
        attrs.sort_unstable();
        attrs.dedup();
        corpus.push(attrs);
    }
    corpus
}

/// Flatten the corpus into a CSR layout: `offsets[node]..offsets[node + 1]` slices `values`.
fn build_csr(corpus: &[Vec<u16>]) -> (Vec<u32>, Vec<u16>) {
    let mut offsets = Vec::with_capacity(corpus.len() + 1);
    let mut values = Vec::new();
    offsets.push(0u32);
    for attrs in corpus {
        values.extend_from_slice(attrs);
        offsets.push(values.len() as u32);
    }
    (offsets, values)
}

/// Invert the corpus into one doc-id bitmap per attribute id.
fn build_posting(corpus: &[Vec<u16>]) -> Vec<RoaringBitmap> {
    let mut posting: Vec<RoaringBitmap> = (0..VOCAB).map(|_| RoaringBitmap::new()).collect();
    for (doc, attrs) in corpus.iter().enumerate() {
        for &id in attrs {
            posting[id as usize].insert(doc as u32);
        }
    }
    posting
}

/// Build the current production index (`HashMap<u32, RoaringTreemap>` via string-keyed
/// attribute encoding), matching what the live filter uses at search time.
fn build_treemap(corpus: &[Vec<u16>]) -> InlineAttributeIndex {
    // Cache one Attribute per label id so we do not rebuild the field string per document.
    let attr_cache: Vec<Attribute> = (0..VOCAB)
        .map(|id| Attribute::from_json_value(&format!("L{id}"), &json!(true)).unwrap())
        .collect();
    let mut index = InlineAttributeIndex::new();
    let mut scratch: Vec<Attribute> = Vec::new();
    for (doc, attrs) in corpus.iter().enumerate() {
        scratch.clear();
        scratch.extend(attrs.iter().map(|&id| attr_cache[id as usize].clone()));
        index.insert_document(doc as u32, &scratch).unwrap();
    }
    index
}

fn build_bitslice(corpus: &[Vec<u16>]) -> InlineAttributeIndexBitslice {
    let attr_cache: Vec<Attribute> = (0..VOCAB)
        .map(|id| Attribute::from_json_value(&format!("L{id}"), &json!(true)).unwrap())
        .collect();
    let mut index = InlineAttributeIndexBitslice::new();
    let mut scratch: Vec<Attribute> = Vec::new();
    for (doc, attrs) in corpus.iter().enumerate() {
        scratch.clear();
        scratch.extend(attrs.iter().map(|&id| attr_cache[id as usize].clone()));
        index.insert_document(doc as u32, &scratch).unwrap();
    }
    index
}

fn bench_live_filter(c: &mut Criterion) {
    let n: usize = std::env::var("LF_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let num_probes: usize = std::env::var("LF_BENCH_PROBES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);

    let mut rng = StdRng::seed_from_u64(0x0DDF_117E_5EED);
    let corpus = build_corpus(n, &mut rng);
    let (offsets, values) = build_csr(&corpus);
    let posting = build_posting(&corpus);
    let frozen = build_treemap(&corpus).freeze();
    let frozen_bitslice = build_bitslice(&corpus).freeze();

    // Random node ids mimic the data-dependent traversal order (defeats prefetching).
    let probes: Vec<u32> = (0..num_probes)
        .map(|_| rng.random_range(0..n as u32))
        .collect();

    let predicates: [(&str, Pred); 3] = [
        ("p1_single_term", Pred::Term(0)),
        ("p2_and2", Pred::And(vec![Pred::Term(0), Pred::Term(1)])),
        (
            "p4_and_or",
            Pred::Or(vec![
                Pred::And(vec![Pred::Term(0), Pred::Term(1)]),
                Pred::And(vec![Pred::Term(2), Pred::Term(3)]),
            ]),
        ),
    ];

    for (name, pred) in &predicates {
        let ast = pred.to_ast();
        let provider: Arc<dyn QueryLabelProvider<u32>> = frozen.make_provider(&ast).unwrap();
        let bitslice_recursive = frozen_bitslice.make_provider(&ast).unwrap();
        let bitslice_dnf = frozen_bitslice.make_dnf_provider(&ast).unwrap();

        // Correctness gate: all three representations must agree on every probe.
        let mut match_count = 0usize;
        for &node in &probes {
            let treemap_hit = provider.is_match(node);
            let recursive_hit = bitslice_recursive.is_match(node);
            let dnf_hit = bitslice_dnf.is_match(node);
            let start = offsets[node as usize] as usize;
            let end = offsets[node as usize + 1] as usize;
            let csr_hit = pred.eval_csr(&values[start..end]);
            let posting_hit = pred.eval_posting(node, &posting);
            assert_eq!(
                treemap_hit, csr_hit,
                "treemap vs csr disagree ({name}, node {node})"
            );
            assert_eq!(
                treemap_hit, posting_hit,
                "treemap vs posting disagree ({name}, node {node})"
            );
            assert_eq!(
                treemap_hit, recursive_hit,
                "treemap vs recursive bitslice disagree ({name}, node {node})"
            );
            assert_eq!(
                treemap_hit, dnf_hit,
                "treemap vs DNF bitslice disagree ({name}, node {node})"
            );
            if treemap_hit {
                match_count += 1;
            }
        }
        let sel_tag = format!(
            "sel={:.1}%",
            match_count as f64 / probes.len() as f64 * 100.0
        );

        let mut group = c.benchmark_group(format!("live_filter/{name}"));
        group.throughput(Throughput::Elements(probes.len() as u64));
        group.sample_size(30);

        group.bench_function(format!("treemap[{sel_tag}]"), |b| {
            b.iter(|| {
                let mut hits = 0u64;
                for &node in &probes {
                    if provider.is_match(black_box(node)) {
                        hits += 1;
                    }
                }
                black_box(hits)
            })
        });

        group.bench_function(format!("csr[{sel_tag}]"), |b| {
            b.iter(|| {
                let mut hits = 0u64;
                for &node in &probes {
                    let start = offsets[node as usize] as usize;
                    let end = offsets[node as usize + 1] as usize;
                    if pred.eval_csr(black_box(&values[start..end])) {
                        hits += 1;
                    }
                }
                black_box(hits)
            })
        });

        group.bench_function(format!("posting[{sel_tag}]"), |b| {
            b.iter(|| {
                let mut hits = 0u64;
                for &node in &probes {
                    if pred.eval_posting(black_box(node), &posting) {
                        hits += 1;
                    }
                }
                black_box(hits)
            })
        });

        group.bench_function(format!("bitslice_recursive[{sel_tag}]"), |b| {
            b.iter(|| {
                let mut hits = 0u64;
                for &node in &probes {
                    if bitslice_recursive.is_match(black_box(node)) {
                        hits += 1;
                    }
                }
                black_box(hits)
            })
        });

        group.bench_function(format!("bitslice_dnf[{sel_tag}]"), |b| {
            b.iter(|| {
                let mut hits = 0u64;
                for &node in &probes {
                    if bitslice_dnf.is_match(black_box(node)) {
                        hits += 1;
                    }
                }
                black_box(hits)
            })
        });

        group.finish();
    }
}

criterion_group!(live_filter_benches, bench_live_filter);
