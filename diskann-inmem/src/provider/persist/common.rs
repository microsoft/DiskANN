/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Read, Seek, SeekFrom, Write};

use diskann_record::{load, save};

use super::super::{Id, Provider};
use crate::{
    ids::IdMap,
    num::{IdLimit, Range},
    repr::Representation,
};

// Compacted representation: live nodes and frozen points, without deleted-node holes.
pub(super) struct Compact {
    frozen_count: u32,
    id_limit: u32,
    // Map from old ID -> new ID for saved nodes.
    renaming: Vec<u32>,
    live_count: u32,
}

impl Compact {
    pub(super) fn renaming(&self) -> &[u32] {
        &self.renaming
    }

    pub(super) fn live_count(&self) -> u32 {
        self.live_count
    }

    pub(super) fn save_external_ids<R>(
        &self,
        provider: &Provider<R, u32>,
        context: save::Context<'_>,
    ) -> save::Result<save::Handle>
    where
        R: Representation,
    {
        let mut external_ids = context.write(Some("external_ids"))?;
        for old in 0..self.id_limit {
            let new = self.renaming[old as usize];
            if new < self.frozen_count || new == u32::MAX {
                continue;
            }
            let external = provider
                .mapping
                .to_external(old)
                .ok_or_else(|| save::Error::message("live node has no external ID"))?;
            external_ids
                .write_all(&external.to_le_bytes())
                .map_err(save::Error::new)?;
        }
        external_ids.finish()
    }
}

pub(super) fn compact<R, M>(provider: &Provider<R, M>, frozen_ids: Range) -> save::Result<Compact>
where
    R: Representation,
    M: Id,
{
    let id_limit = provider.representation.id_limit().value();
    if frozen_ids.end() > id_limit {
        return Err(save::Error::message("invalid frozen-point range"));
    }
    let frozen_count = frozen_ids.end() - frozen_ids.start();
    let mut renaming = Vec::new();
    renaming
        .try_reserve_exact(id_limit as usize)
        .map_err(save::Error::new)?;
    renaming.resize(id_limit as usize, u32::MAX);
    let mut next = frozen_count;
    for old in 0..id_limit {
        let readable = provider.representation.is_readable(old) == Some(true);
        let external = provider.mapping.to_external(old);
        if frozen_ids.iter().contains(&old) {
            if !readable || external.is_some() {
                return Err(save::Error::message(
                    "invalid frozen-point state or mapping",
                ));
            }
            renaming[old as usize] = old - frozen_ids.start();
        } else {
            match (readable, external) {
                (true, Some(external)) => {
                    if provider.mapping.to_internal(&external) != Some(old) {
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
    if provider.mapping.len() != live_count as usize {
        return Err(save::Error::message(
            "live-node and external-ID counts disagree",
        ));
    }
    Ok(Compact {
        frozen_count,
        id_limit,
        renaming,
        live_count,
    })
}

pub(super) fn load_external_ids(
    object: &load::Object<'_>,
    live_count: u32,
    frozen_count: u32,
    id_limit: IdLimit,
) -> load::Result<IdMap<u32>> {
    if frozen_count.checked_add(live_count) > Some(id_limit.value()) {
        return Err(load::Error::message(
            "external-ID count exceeds the provider ID limit",
        ));
    }
    let expected = u64::from(live_count) * size_of::<u32>() as u64;
    let handle: save::Handle = object.field("external_ids")?;
    let mut reader = object.read(&handle)?;
    let actual = reader.seek(SeekFrom::End(0)).map_err(load::Error::new)?;
    if actual != expected {
        return Err(load::Error::message(format!(
            "external_ids artifact has {actual} bytes, expected {expected}",
        )));
    }
    reader.rewind().map_err(load::Error::new)?;

    let mapping = IdMap::new(id_limit);
    for offset in 0..live_count {
        let mut bytes = [0; size_of::<u32>()];
        reader.read_exact(&mut bytes).map_err(load::Error::new)?;
        mapping
            .insert(u32::from_le_bytes(bytes), frozen_count + offset)
            .map_err(load::Error::new)?;
    }
    Ok(mapping)
}
