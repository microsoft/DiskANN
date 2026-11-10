/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann::graph::AdjacencyList;
use diskann_record::save;

use super::Full;

impl Full<f32> {
    pub(crate) fn frozen_count(&self) -> u32 {
        self.start_point_ids.end()
    }

    pub(crate) fn save(
        &self,
        context: save::Context<'_>,
        renaming: &[u32],
        live_count: u32,
    ) -> save::Result<save::Record<'_>> {
        let reader = self.reader().map_err(save::Error::new)?;
        let mut vectors = context.write(Some("vectors"))?;
        let mut graph = context.write(Some("graph"))?;
        let mut neighbors = AdjacencyList::new();
        for (old, new) in renaming.iter().enumerate() {
            if *new == u32::MAX {
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
            }
            neighbors.retain(|target| renaming[*target as usize] != u32::MAX);
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
            ("live_count", u64::from(live_count)),
            ("frozen_count", u64::from(self.frozen_count())),
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
        Ok(record)
    }
}
