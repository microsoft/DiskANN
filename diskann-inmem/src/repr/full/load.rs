/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Read, Seek, SeekFrom};

use diskann_record::{load, save};

use super::{DEFAULT_LOOKAHEAD, Full};
use crate::{
    num::{Bytes, Capacity, IdLimit, MaxDegree, Range},
    store::{self, Store, intrusive::Intrusive},
    tag::AtomicTag,
};

impl Full<f32> {
    pub(crate) fn load(
        object: &load::Object<'_>,
        capacity: Option<Capacity>,
        config: store::Config,
    ) -> load::Result<Self> {
        diskann_record::load_fields!(object, [
            live_count: u32,
            frozen_count: u32,
            dim: usize,
            max_degree: u32,
            metric: &str,
        ]);
        let metric = metric.parse().map_err(load::Error::new)?;
        let capacity = capacity.unwrap_or(Capacity::new(live_count as usize));
        if capacity.value() < live_count as usize {
            return Err(load::Error::message(
                "capacity is smaller than the saved live-node count",
            ));
        }

        let id_limit = frozen_count
            .checked_add(u32::try_from(capacity.value()).map_err(load::Error::new)?)
            .map(IdLimit::new)
            .ok_or_else(|| load::Error::message("capacity plus frozen points exceeds u32::MAX"))?;
        let saved_count = frozen_count
            .checked_add(live_count)
            .ok_or_else(|| load::Error::message("saved point count exceeds u32::MAX"))?;
        let bytes = dim
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| load::Error::message("vector dimension overflows usize"))?;
        let vector_bytes = (saved_count as u64)
            .checked_mul(u64::try_from(bytes).map_err(load::Error::new)?)
            .ok_or_else(|| load::Error::message("vector artifact size overflows u64"))?;
        let mut vectors = artifact(object, "vectors", vector_bytes)?;
        let graph_handle: save::Handle = object.field("graph")?;
        let mut graph = object.read(&graph_handle)?;
        let graph_bytes = graph.seek(SeekFrom::End(0)).map_err(load::Error::new)?;
        graph.rewind().map_err(load::Error::new)?;
        let min_graph_bytes = u64::from(saved_count) * size_of::<u32>() as u64;
        let max_graph_bytes = min_graph_bytes
            .checked_mul(u64::from(max_degree) + 1)
            .ok_or_else(|| load::Error::message("graph artifact size overflows u64"))?;
        if !(min_graph_bytes..=max_graph_bytes).contains(&graph_bytes)
            || !graph_bytes.is_multiple_of(size_of::<u32>() as u64)
        {
            return Err(load::Error::message("invalid graph artifact length"));
        }

        let vector_stride = Bytes::new(bytes)
            .checked_add(AtomicTag::SIZE)
            .and_then(|bytes| bytes.checked_next_multiple_of(Bytes::CACHELINE))
            .ok_or_else(|| load::Error::message("vector slot size overflows usize"))?;
        let graph_stride = (max_degree as usize)
            .checked_add(1)
            .and_then(|degree| degree.checked_mul(size_of::<u32>()))
            .ok_or_else(|| load::Error::message("graph slot size overflows usize"))?;
        for stride in [vector_stride.value(), graph_stride] {
            let size = id_limit
                .as_usize()
                .checked_mul(stride)
                .ok_or_else(|| load::Error::message("store allocation size overflows usize"))?;
            std::alloc::Layout::from_size_align(size, crate::num::Align::_128.value())
                .map_err(load::Error::new)?;
        }

        let layout = store::Layout::new(id_limit, MaxDegree::new(max_degree as usize));
        let store = Store::new(layout, config, Intrusive::config(Bytes::new(bytes)))
            .map_err(load::Error::new)?;
        let mut neighbors = Vec::new();
        for id in 0..saved_count {
            let mut slot = store
                .slot(id)
                .ok_or_else(|| load::Error::message("could not acquire a restored slot"))?;
            let data = slot.data().as_mut_slice();
            vectors.read_exact(data).map_err(load::Error::new)?;
            if cfg!(target_endian = "big") {
                for value in data.chunks_exact_mut(size_of::<f32>()) {
                    value.reverse();
                }
            }

            let degree = read_u32(&mut graph)?;
            if degree > max_degree {
                return Err(load::Error::message(
                    "saved neighbor count exceeds max_degree",
                ));
            }
            neighbors.clear();
            neighbors
                .try_reserve(degree as usize)
                .map_err(load::Error::new)?;
            for _ in 0..degree {
                let target = read_u32(&mut graph)?;
                if target >= saved_count {
                    return Err(load::Error::message("saved edge targets an unsaved slot"));
                }
                neighbors.push(target);
            }
            store
                .neighbors()
                .set(id, &neighbors)
                .map_err(load::Error::new)?;
            if id < frozen_count {
                slot.freeze();
            } else {
                slot.publish();
            }
        }
        if graph.stream_position().map_err(load::Error::new)? != graph_bytes {
            return Err(load::Error::message(
                "graph artifact contains extra rows or edges",
            ));
        }
        Ok(Self {
            store,
            start_point_ids: Range::start_plus(0, frozen_count)
                .ok_or_else(|| load::Error::message("invalid frozen-point range"))?,
            capacity,
            metric,
            lookahead: Some(DEFAULT_LOOKAHEAD),
            _type: std::marker::PhantomData,
        })
    }
}

pub(crate) fn artifact<'a>(
    object: &'a load::Object<'_>,
    key: &str,
    expected: u64,
) -> load::Result<load::Reader<'a>> {
    let handle: save::Handle = object.field(key)?;
    let mut reader = object.read(&handle)?;
    let actual = reader.seek(SeekFrom::End(0)).map_err(load::Error::new)?;
    if actual != expected {
        return Err(load::Error::message(format!(
            "{} artifact has {} bytes, expected {}",
            key, actual, expected,
        )));
    }
    reader.rewind().map_err(load::Error::new)?;
    Ok(reader)
}

pub(crate) fn read_u32(reader: &mut impl Read) -> load::Result<u32> {
    let mut bytes = [0; size_of::<u32>()];
    reader.read_exact(&mut bytes).map_err(load::Error::new)?;
    Ok(u32::from_le_bytes(bytes))
}
