/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Convert a TSV with embeddings + multi-label rows to the DiskANN binary +
//! JSONL label files used by `compute_groundtruth` and the filtered search
//! benchmark.
//!
//! Expected TSV columns (tab-separated):
//!   col 0: embedding, space-separated integer values (parsed as `u8`)
//!   col 1: ignored
//!   col 2: comma-separated label codes (e.g. `AT,CH,DE`)
//!
//! Output files:
//!   --base-out             : DiskANN u8 binary of all rows
//!   --queries-out          : DiskANN u8 binary of the first --num-queries rows
//!   --base-labels-out      : JSONL, one `Document` per row
//!                            `{"doc_id": i, "<code1>": true, "<code2>": true, ...}`
//!   --query-predicates-out : JSONL, one `QueryExpression` per query, all with
//!                            the same predicate
//!
//! Predicate sources (mutually exclusive):
//!   --query-label   <CODE>     : shorthand for `{"<CODE>": {"$eq": true}}`
//!   --query-filter-json <JSON> : arbitrary filter JSON, e.g.
//!                                `{"$and":[{"MG":{"$eq":true}},{"PT":{"$eq":true}}]}`
//!
//! The presence-flag encoding is required because the label-filter evaluator
//! uses strict `serde_json::Value` equality (`$in`/`$nin` were removed), so
//! `$eq` against a JSON array does NOT do set-membership matching.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use diskann_utils::io::Metadata;

#[derive(Parser, Debug)]
#[command(
    name = "tsv_to_diskann_bin",
    about = "Convert label-tagged TSV embeddings into DiskANN bin + JSONL label files"
)]
struct Args {
    /// Input TSV file.
    #[arg(long)]
    input: PathBuf,

    /// Output base vectors (DiskANN u8 binary).
    #[arg(long)]
    base_out: PathBuf,

    /// Output query vectors (DiskANN u8 binary), the first `--num-queries` rows.
    #[arg(long)]
    queries_out: PathBuf,

    /// Output base labels (JSONL Documents).
    #[arg(long)]
    base_labels_out: PathBuf,

    /// Output per-query predicates (JSONL QueryExpressions). All queries use
    /// the same predicate, taken from either `--query-label` or
    /// `--query-filter-json`.
    #[arg(long)]
    query_predicates_out: PathBuf,

    /// Number of queries to extract from the head of the file.
    #[arg(long, default_value_t = 1000)]
    num_queries: usize,

    /// Single-label shorthand: every query filters on `{<label>: {"$eq": true}}`.
    /// Mutually exclusive with `--query-filter-json`.
    #[arg(long, conflicts_with = "query_filter_json")]
    query_label: Option<String>,

    /// Raw JSON predicate, applied identically to every query. Use this for
    /// multi-label or boolean expressions, e.g.
    /// `{"$and":[{"MG":{"$eq":true}},{"PT":{"$eq":true}}]}`.
    /// Mutually exclusive with `--query-label`.
    #[arg(long, conflicts_with = "query_label")]
    query_filter_json: Option<String>,

    /// Skip emitting the base bin and base label JSONL. Use this when you have
    /// already produced those files for the same TSV and only want to refresh
    /// the queries / per-query predicates for a different filter.
    #[arg(long, default_value_t = false)]
    queries_only: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate and resolve the query predicate up front.
    if args.query_label.is_none() && args.query_filter_json.is_none() {
        bail!("either --query-label or --query-filter-json must be provided");
    }
    let predicate_json: String = if let Some(label) = &args.query_label {
        if !is_safe_key(label) {
            bail!(
                "--query-label `{}` contains characters not allowed in a JSON field name",
                label
            );
        }
        format!("{{\"{}\":{{\"$eq\":true}}}}", label)
    } else {
        args.query_filter_json.as_ref().unwrap().trim().to_string()
    };

    let input =
        File::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;
    let reader = BufReader::new(input);

    // Optionally skip the base outputs. We still need a sink for them so the
    // loop stays branch-free; use a tiny in-memory throwaway when skipping.
    let mut base_writer: Box<dyn WriteSeek> = if args.queries_only {
        Box::new(SinkWithSeek)
    } else {
        Box::new(BufWriter::new(File::create(&args.base_out).with_context(
            || format!("creating {}", args.base_out.display()),
        )?))
    };
    let mut queries_writer = BufWriter::new(
        File::create(&args.queries_out)
            .with_context(|| format!("creating {}", args.queries_out.display()))?,
    );
    let mut base_labels_writer: Box<dyn Write> = if args.queries_only {
        Box::new(std::io::sink())
    } else {
        Box::new(BufWriter::new(
            File::create(&args.base_labels_out)
                .with_context(|| format!("creating {}", args.base_labels_out.display()))?,
        ))
    };
    let mut query_pred_writer = BufWriter::new(
        File::create(&args.query_predicates_out)
            .with_context(|| format!("creating {}", args.query_predicates_out.display()))?,
    );

    // Reserve room for the 8-byte header on both bin files; we rewrite it after
    // the final count is known (matching the pattern in subsample_bin.rs).
    let placeholder = Metadata::new(0u32, 0u32)?;
    placeholder.write(&mut base_writer)?;
    placeholder.write(&mut queries_writer)?;

    let mut dim: Option<usize> = None;
    let mut row_buf: Vec<u8> = Vec::new();
    let mut npoints: u32 = 0;
    let mut nqueries: u32 = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_num + 1))?;
        if line.is_empty() {
            continue;
        }

        let mut cols = line.split('\t');
        let embedding_str = cols
            .next()
            .ok_or_else(|| anyhow!("line {}: missing embedding column", line_num + 1))?;
        let _ignored = cols.next();
        let labels_str = cols
            .next()
            .ok_or_else(|| anyhow!("line {}: missing label column", line_num + 1))?;

        // Parse embedding values.
        row_buf.clear();
        for tok in embedding_str.split_ascii_whitespace() {
            let v: u8 = tok
                .parse()
                .with_context(|| format!("line {}: invalid u8 token `{}`", line_num + 1, tok))?;
            row_buf.push(v);
        }

        match dim {
            None => dim = Some(row_buf.len()),
            Some(d) if d != row_buf.len() => {
                bail!(
                    "line {}: dim mismatch (expected {}, got {})",
                    line_num + 1,
                    d,
                    row_buf.len()
                );
            }
            _ => {}
        }

        // Write base vector (or discard if --queries-only).
        base_writer.write_all(&row_buf)?;

        // Write query vector if within the head slice.
        if (npoints as usize) < args.num_queries {
            queries_writer.write_all(&row_buf)?;
            nqueries += 1;
        }

        // Write the base label as a JSONL Document with a presence flag per code
        // (no-op when --queries-only).
        write!(base_labels_writer, "{{\"doc_id\":{}", npoints)?;
        for raw in labels_str.split(',') {
            let code = raw.trim();
            if code.is_empty() {
                continue;
            }
            if !is_safe_key(code) {
                bail!(
                    "line {}: label code `{}` contains characters not allowed in a JSON field name",
                    line_num + 1,
                    code
                );
            }
            write!(base_labels_writer, ",\"{}\":true", code)?;
        }
        writeln!(base_labels_writer, "}}")?;

        npoints = npoints
            .checked_add(1)
            .ok_or_else(|| anyhow!("too many rows (>{})", u32::MAX))?;

        // When we only want queries, stop reading the TSV once we've gathered
        // enough query rows.
        if args.queries_only && (nqueries as usize) >= args.num_queries {
            break;
        }
    }

    let dim = dim.ok_or_else(|| anyhow!("input is empty"))?;

    if (nqueries as usize) < args.num_queries {
        bail!(
            "requested {} queries but input only has {} rows",
            args.num_queries,
            nqueries
        );
    }

    // Rewrite the bin headers with the real counts.
    use std::io::Seek;
    if !args.queries_only {
        base_writer.flush()?;
        base_writer.seek(std::io::SeekFrom::Start(0))?;
        Metadata::new(npoints, dim as u32)?.write(&mut base_writer)?;
    }

    queries_writer.flush()?;
    queries_writer.get_mut().seek(std::io::SeekFrom::Start(0))?;
    Metadata::new(nqueries, dim as u32)?.write(&mut queries_writer)?;

    // Emit the per-query predicates: identical filter for every query.
    for q in 0..nqueries {
        writeln!(
            query_pred_writer,
            "{{\"query_id\":{},\"filter\":{}}}",
            q, predicate_json
        )?;
    }

    base_writer.flush()?;
    queries_writer.flush()?;
    base_labels_writer.flush()?;
    query_pred_writer.flush()?;

    println!(
        "rows={} dim={} queries={} predicate={} queries_only={}",
        npoints, dim, nqueries, predicate_json, args.queries_only
    );
    Ok(())
}

/// Restrict label codes to characters that are safe to embed unquoted in a JSON
/// field name. The country-code style labels in the target TSV (e.g. `MG`,
/// `AT`) are all in this set; this guards against malformed input.
fn is_safe_key(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Trait alias used so the base writer can be either a real file or a discard
/// sink in --queries-only mode while still satisfying `Write + Seek`.
trait WriteSeek: Write + std::io::Seek {}
impl<T: Write + std::io::Seek> WriteSeek for T {}

/// `io::sink()` does not implement `Seek`. Provide a tiny wrapper that
/// discards all writes and ignores seeks, used only when the base output is
/// suppressed.
struct SinkWithSeek;
impl Write for SinkWithSeek {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl std::io::Seek for SinkWithSeek {
    fn seek(&mut self, _pos: std::io::SeekFrom) -> std::io::Result<u64> {
        Ok(0)
    }
}
