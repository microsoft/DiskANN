/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Read, Seek, SeekFrom, Write};

use diskann_record::{load, save};

use super::super::Provider;
use crate::{
    ids::IdMap,
    num::IdLimit,
    repr::{Representation, full::Compaction},
};

pub(super) fn save_external_ids<R>(
    provider: &Provider<R, u32>,
    compaction: &Compaction,
    context: save::Context<'_>,
) -> save::Result<save::Handle>
where
    R: Representation,
{
    let mut external_ids = context.write(Some("external_ids"))?;
    let original_frozen_ids = compaction.original_frozen_ids();
    let mut saved_live = 0;
    for (old, &new) in compaction.renaming().iter().enumerate() {
        let old = u32::try_from(old).map_err(save::Error::new)?;
        let external = provider.mapping.to_external(old);
        if new == u32::MAX {
            if external.is_some() {
                return Err(save::Error::message("live slots and the ID map disagree"));
            }
            continue;
        }
        if original_frozen_ids.iter().contains(&old) {
            if external.is_some() {
                return Err(save::Error::message("invalid frozen-point mapping"));
            }
            continue;
        }
        let external =
            external.ok_or_else(|| save::Error::message("live node has no external ID"))?;
        if provider.mapping.to_internal(&external) != Some(old) {
            return Err(save::Error::message("inconsistent external-ID mapping"));
        }
        external_ids
            .write_all(&external.to_le_bytes())
            .map_err(save::Error::new)?;
        saved_live += 1;
    }
    if saved_live != compaction.live_count() {
        return Err(save::Error::message(
            "live-node and external-ID counts disagree",
        ));
    }
    if provider.mapping.len() != saved_live as usize {
        return Err(save::Error::message(
            "live-node and external-ID counts disagree",
        ));
    }
    external_ids.finish()
}

pub(super) fn load_external_ids(
    object: &load::Object<'_>,
    live_count: u32,
    frozen_ids: crate::num::Range,
    id_limit: IdLimit,
) -> load::Result<IdMap<u32>> {
    if frozen_ids
        .len()
        .checked_add(live_count as usize)
        .and_then(|count| u32::try_from(count).ok())
        > Some(id_limit.value())
    {
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
    let saved_count = frozen_ids
        .len()
        .checked_add(live_count as usize)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| load::Error::message("saved point count exceeds u32::MAX"))?;
    for internal in 0..saved_count {
        if frozen_ids.iter().contains(&internal) {
            continue;
        }
        let mut bytes = [0; size_of::<u32>()];
        reader.read_exact(&mut bytes).map_err(load::Error::new)?;
        mapping
            .insert(u32::from_le_bytes(bytes), internal)
            .map_err(load::Error::new)?;
    }
    Ok(mapping)
}
