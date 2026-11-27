/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann::graph::AdjacencyList;
use diskann_record::save;

use super::Full;
use crate::{epoch, num::Range, repr::Representation};

pub(crate) struct Compaction {
    frozen_ids: Range,
    live_count: u32,
    original_frozen_ids: Range,
    renaming: Vec<u32>,
}

impl Compaction {
    pub(crate) fn frozen_ids(&self) -> Range {
        self.frozen_ids
    }

    pub(crate) fn live_count(&self) -> u32 {
        self.live_count
    }

    pub(crate) fn original_frozen_ids(&self) -> Range {
        self.original_frozen_ids
    }

    pub(crate) fn renaming(&self) -> &[u32] {
        &self.renaming
    }
}

impl Full<f32> {
    pub(crate) fn save(
        &self,
        context: save::Context<'_>,
        _writers: &epoch::WriterExclusionGuard<'_>,
    ) -> save::Result<(save::Record<'_>, Compaction)> {
        let compaction = self.compaction()?;
        let reader = self.reader(None).map_err(save::Error::new)?;
        let mut vectors = context.write(Some("vectors"))?;
        let mut graph = context.write(Some("graph"))?;
        let mut neighbors = AdjacencyList::new();
        let renaming = compaction.renaming();
        for (old, &new) in renaming.iter().enumerate() {
            if new == u32::MAX {
                continue;
            }
            let old = u32::try_from(old).map_err(save::Error::new)?;
            let data = reader
                .read(old)
                .ok_or_else(|| save::Error::message("saved slot is not readable"))?;
            if cfg!(target_endian = "little") {
                vectors.write_all(data).map_err(save::Error::new)?;
            } else {
                for value in data.chunks_exact(size_of::<f32>()) {
                    vectors
                        .write_all(&[value[3], value[2], value[1], value[0]])
                        .map_err(save::Error::new)?;
                }
            }

            self.store
                .neighbors()
                .get(old, &mut neighbors)
                .map_err(save::Error::new)?;
            for target in neighbors.iter() {
                if *target as usize >= renaming.len() {
                    return Err(save::Error::message("graph edge is outside the provider"));
                }
                if renaming[*target as usize] == u32::MAX {
                    return Err(save::Error::message(format!(
                        "graph edge {old} -> {target} targets an omitted slot; consolidate all live nodes and frozen points before saving",
                    )));
                }
            }
            let degree = u32::try_from(neighbors.len()).map_err(save::Error::new)?;
            graph
                .write_all(&degree.to_le_bytes())
                .map_err(save::Error::new)?;
            for target in neighbors.iter() {
                graph
                    .write_all(&renaming[*target as usize].to_le_bytes())
                    .map_err(save::Error::new)?;
            }
        }

        let mut record = save::Record::empty();
        for (key, value) in [
            ("live_count", u64::from(compaction.live_count())),
            ("frozen_start", u64::from(compaction.frozen_ids().start())),
            ("frozen_count", self.start_point_ids.len() as u64),
            ("dim", u64::try_from(self.dim()).map_err(save::Error::new)?),
            (
                "max_degree",
                u64::from(self.store.neighbors().max_degree_u32()),
            ),
        ] {
            record.insert(
                key,
                save::Saveable::save(&value, context.clone())?.into_owned(),
            )?;
        }
        record.insert("metric", save::Value::String(self.metric.as_str().into()))?;
        record.insert("vectors", vectors.finish()?)?;
        record.insert("graph", graph.finish()?)?;
        Ok((record, compaction))
    }

    fn compaction(&self) -> save::Result<Compaction> {
        let id_limit = self.store.id_limit().value();
        if self.start_point_ids.end() > id_limit {
            return Err(save::Error::message("invalid frozen-point range"));
        }

        let mut renaming = Vec::new();
        renaming
            .try_reserve_exact(id_limit as usize)
            .map_err(save::Error::new)?;
        renaming.resize(id_limit as usize, u32::MAX);

        let mut next = 0u32;
        let mut frozen_start = None;
        for old in 0..id_limit {
            if old == self.start_point_ids.start() {
                frozen_start = Some(next);
            }
            if self.is_readable(old) == Some(true) {
                renaming[old as usize] = next;
                next = next
                    .checked_add(1)
                    .ok_or_else(|| save::Error::message("saved point count exceeds u32::MAX"))?;
            }
        }
        let frozen_start = frozen_start.unwrap_or(next);
        for old in self.start_point_ids.iter() {
            if renaming[old as usize] == u32::MAX {
                return Err(save::Error::message("frozen point is not readable"));
            }
        }

        let frozen_count = u32::try_from(self.start_point_ids.len()).map_err(save::Error::new)?;
        let live_count = next
            .checked_sub(frozen_count)
            .ok_or_else(|| save::Error::message("saved point count is inconsistent"))?;
        let frozen_ids = Range::start_plus(frozen_start, frozen_count)
            .ok_or_else(|| save::Error::message("compacted frozen-point range overflows"))?;
        Ok(Compaction {
            frozen_ids,
            live_count,
            original_frozen_ids: self.start_point_ids,
            renaming,
        })
    }
}
