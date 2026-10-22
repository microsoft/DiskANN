/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Parent capture is opt-in on a separate diagnostic build. Timed static jobs
//! never enable it; replay preserves captured point IDs and logical row order.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use diskann::ivf::workingset::{ListView, PointRef};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    Benchmark, Checkpoint, Output,
};
use diskann_utils::views::rowmajor::Matrix;
use serde::Serialize;

use super::{run_workload, Data, WorkloadResults};
use crate::{
    graph_ivf::{
        build::build_with_store,
        data::{load_data, load_vectors, validate_vectors},
        provider::Store,
    },
    inputs::graph_ivf_two_means::{GraphIvfParentReplay, GraphIvfTwoMeans, Layout},
};

pub(in crate::graph_ivf) struct Capture {
    directory: PathBuf,
    selected: Vec<usize>,
    seen: usize,
    snapshots: Vec<Snapshot>,
}

#[derive(Serialize, Clone)]
struct Snapshot {
    ordinal: usize,
    list_id: u32,
    inserted_so_far: usize,
    rows: usize,
    dimensions: usize,
    coordinates: PathBuf,
    ids_file: PathBuf,
}

impl Capture {
    pub(in crate::graph_ivf) fn record(
        &mut self,
        lists: &[ListView<'_, u32, u32>],
        inserted_so_far: usize,
    ) -> anyhow::Result<()> {
        for list in lists {
            let ordinal = self.seen;
            self.seen += 1;
            if self.selected.binary_search(&ordinal).is_err() {
                continue;
            }
            let dim = list
                .points
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty split parent"))?
                .vector
                .len();
            let coordinates = self.directory.join(format!("parent-{ordinal:06}.fbin"));
            let ids_file = self.directory.join(format!("parent-{ordinal:06}-ids.json"));
            let mut file = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&coordinates)?,
            );
            file.write_all(&u32::try_from(list.points.len())?.to_le_bytes())?;
            file.write_all(&u32::try_from(dim)?.to_le_bytes())?;
            for point in &list.points {
                for &value in point.vector {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            file.flush()?;
            let ids: Vec<_> = list.points.iter().map(|point| point.id).collect();
            let mut ids_output = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&ids_file)?,
            );
            serde_json::to_writer(&mut ids_output, &ids)?;
            ids_output.flush()?;
            self.snapshots.push(Snapshot {
                ordinal,
                list_id: list.id,
                inserted_so_far,
                rows: list.points.len(),
                dimensions: dim,
                coordinates,
                ids_file,
            });
        }
        Ok(())
    }
}

pub(in crate::graph_ivf) struct Replay;

#[derive(Serialize)]
pub(in crate::graph_ivf) struct Results {
    source: &'static str,
    capture_timing: &'static str,
    replay_seeding: &'static str,
    requested_ordinals: Vec<usize>,
    parents_seen: usize,
    missing_ordinals: Vec<usize>,
    parents: Vec<Parent>,
}

#[derive(Serialize)]
struct Parent {
    snapshot: Snapshot,
    workloads: Vec<WorkloadResults>,
}

impl Benchmark for Replay {
    type Input = GraphIvfParentReplay;
    type Output = Results;
    fn try_match(&self, _: &Self::Input, context: &MatchContext) -> Score {
        context.success(0)
    }
    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Diagnostic baseline build and replay of predetermined split parents"
        )
    }
    fn run(
        &self,
        input: &Self::Input,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Results> {
        let (data, _) = load_data(&input.index)?;
        input.index.build.validate_rows(data.nrows())?;
        validate_vectors(data.as_view(), "parent replay dataset")?;
        std::fs::create_dir(&input.directory)?;
        let capture = Arc::new(Mutex::new(Capture {
            directory: input.directory.clone(),
            selected: input.parent_ordinals.clone(),
            seen: 0,
            snapshots: Vec::new(),
        }));
        let mut store = Store::new(data.ncols());
        store.capture = Some(Arc::clone(&capture));
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        writeln!(
            output,
            "Capturing predetermined parents in an independent, untimed baseline build"
        )?;
        let (index, graph, _) =
            runtime.block_on(build_with_store(&input.index, data.as_view(), store))?;
        index.provider().validate_membership(data.nrows())?;
        drop(index);
        drop(graph);
        drop(data);
        let capture = Arc::try_unwrap(capture)
            .map_err(|_| anyhow::anyhow!("capture still borrowed"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("parent capture lock poisoned"))?;
        // Write the capture manifest before replay so an interrupted diagnostic run
        // still has an auditable mapping of every exported parent and row order.
        serde_json::to_writer_pretty(
            File::create(input.directory.join("manifest.json"))?,
            &capture.snapshots,
        )?;
        let missing_ordinals = input
            .parent_ordinals
            .iter()
            .copied()
            .filter(|n| *n >= capture.seen)
            .collect();
        let mut parents = Vec::new();
        for snapshot in capture.snapshots {
            let matrix = load_vectors(
                &snapshot.coordinates,
                crate::inputs::graph_ivf::FileType::Float32,
            )?;
            let ids: Vec<usize> = serde_json::from_reader(File::open(&snapshot.ids_file)?)?;
            anyhow::ensure!(
                ids.len() == matrix.nrows(),
                "snapshot ID count differs from rows"
            );
            let config = GraphIvfTwoMeans {
                strategies: crate::inputs::graph_ivf_two_means::FitStrategy::all(),
                order: crate::inputs::graph_ivf_two_means::FitOrder::Rotating,
                data: None,
                data_type: crate::inputs::graph_ivf::FileType::Float32,
                bucket_sizes: vec![NonZeroUsize::new(snapshot.rows)
                    .ok_or_else(|| anyhow::anyhow!("empty snapshot"))?],
                dimensions: vec![NonZeroUsize::new(snapshot.dimensions)
                    .ok_or_else(|| anyhow::anyhow!("zero dimension"))?],
                iterations: vec![input.index.build.two_means_iterations],
                layouts: vec![Layout::Contiguous, Layout::Scattered],
                seed: input.seed,
                repetitions: input.repetitions,
                warmup_repetitions: input.warmup_repetitions,
            };
            let mut workloads = Vec::new();
            for layout in &config.layouts {
                let values = matrix.as_slice().to_vec();
                let data = Data::from_values(
                    *layout,
                    snapshot.rows,
                    snapshot.dimensions,
                    input.seed,
                    values,
                );
                let points: Vec<_> = data
                    .points()
                    .into_iter()
                    .zip(&ids)
                    .map(|(p, &id)| PointRef {
                        id,
                        vector: p.vector,
                    })
                    .collect();
                workloads.push(run_workload(
                    &config,
                    *layout,
                    &points,
                    snapshot.dimensions,
                    input.index.build.two_means_iterations.get(),
                    output,
                )?);
            }
            parents.push(Parent {
                snapshot,
                workloads,
            });
        }
        Ok(Results {
            source: "actual_split_parents",
            capture_timing: "excluded; capture build timings discarded",
            replay_seeding: "same fresh seed per fit, not the original build RNG trajectory",
            requested_ordinals: input.parent_ordinals.clone(),
            parents_seen: capture.seen,
            missing_ordinals,
            parents,
        })
    }
}
