/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_record::{Version, save};

use super::Provider;
use crate::repr::{Full, Representation};

/// Save a compact snapshot without modifying the provider's internal IDs.
///
/// All mutations must be paused and drained until this operation returns. Searches
/// may continue. Temporary storage is one `u32` per internal slot and one graph row,
/// in addition to the artifact writers' buffers.
impl save::Save for Provider<Full<f32>, u32> {
    const VERSION: Version = super::load::VERSION;

    fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
        let representation = &self.representation;
        let frozen_count = representation.frozen_count();
        let id_limit = representation.id_limit().value();
        let mut renaming = Vec::new();
        renaming
            .try_reserve_exact(id_limit as usize)
            .map_err(save::Error::new)?;
        renaming.resize(id_limit as usize, u32::MAX);
        let mut next = frozen_count;
        for old in 0..id_limit {
            let readable = representation.is_readable(old) == Some(true);
            let external = self.mapping.to_external(old);
            if old < frozen_count {
                if !readable || external.is_some() {
                    return Err(save::Error::message(
                        "invalid frozen-point state or mapping",
                    ));
                }
                renaming[old as usize] = old;
            } else {
                match (readable, external) {
                    (true, Some(external)) => {
                        if self.mapping.to_internal(&external) != Some(old) {
                            return Err(save::Error::message("inconsistent external-ID mapping"));
                        }
                        renaming[old as usize] = next;
                        next += 1;
                    }
                    (false, None) => {}
                    _ => return Err(save::Error::message("live slots and the ID map disagree")),
                }
            }
        }
        let live_count = next - frozen_count;
        if self.mapping.len() != live_count as usize {
            return Err(save::Error::message(
                "live-node and external-ID counts disagree",
            ));
        }

        let mut record = representation.save(context.clone(), &renaming, live_count)?;
        let mut external_ids = context.write(Some("external_ids"))?;
        for old in frozen_count..id_limit {
            if renaming[old as usize] == u32::MAX {
                continue;
            }
            let external = self
                .mapping
                .to_external(old)
                .ok_or_else(|| save::Error::message("live node has no external ID"))?;
            external_ids
                .write_all(&external.to_le_bytes())
                .map_err(save::Error::new)?;
        }
        record.insert("external_ids", external_ids.finish()?)?;
        Ok(record)
    }
}
