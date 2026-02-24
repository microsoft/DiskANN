/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Instant,
};

use bincode;
use bit_set::BitSet;
use diskann_providers::utils::{ParallelIteratorInPool, RayonThreadPool};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::utils::MultiLabel;

// Helper struct for serializing BitSet as Vec<u8> (raw storage)
#[derive(Serialize, Deserialize)]
struct SerializableBitSet(Vec<u8>);

impl From<&BitSet> for SerializableBitSet {
    fn from(bs: &BitSet) -> Self {
        SerializableBitSet(bs.get_ref().to_bytes())
    }
}

impl From<SerializableBitSet> for BitSet {
    fn from(val: SerializableBitSet) -> Self {
        BitSet::from_bytes(&val.0)
    }
}

fn read_lines<P>(filename: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let buf_reader = BufReader::new(file);
    buf_reader.lines().collect()
}

// read labels from file and convert them to bitmaps
// for base lable, each row of the file is the label corresponding to the points in the index, each label is a key value pair
//   example:     CAT=ExteriorAccessories, CAT=ReplacementParts, RATING=5
// for query label, each row is the label corresponding to the query, each label is a key value pair separated by "&" or '|'
//  example:     CAT=ExteriorAccessories&RATING=4|RATING=5
pub fn read_labels_to_bitmap(
    query_label_filename: &str,
    base_label_filename: &str,
    pool: &RayonThreadPool,
) -> Result<Vec<BitSet>, io::Error> {
    let cache_filename = format!("{}.bitmap_cache.bin", query_label_filename);
    if Path::new(&cache_filename).exists() {
        info!("Loading cached bitmaps from: {}", cache_filename);
        let file = File::open(&cache_filename)?;
        let reader = BufReader::new(file);
        let ser_bitmaps: Vec<SerializableBitSet> =
            bincode::deserialize_from(reader).map_err(io::Error::other)?;
        return Ok(ser_bitmaps.into_iter().map(|s| s.into()).collect());
    }
    info!("Reading query labels from: {}", query_label_filename);
    info!("Reading base labels from: {}", base_label_filename);
    let query_strings = read_lines(query_label_filename)?;
    let metadata_strings = read_lines(base_label_filename)?;
    let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, pool);
    // Save to cache
    let file = File::create(&cache_filename)?;
    let writer = std::io::BufWriter::new(file);
    let ser_bitmaps: Vec<SerializableBitSet> =
        bitmaps.iter().map(SerializableBitSet::from).collect();
    bincode::serialize_into(writer, &ser_bitmaps).map_err(io::Error::other)?;
    Ok(bitmaps)
}

pub fn process_bitmap_for_labels(
    query_strings: Vec<String>,
    metadata_strings: Vec<String>,
    pool: &RayonThreadPool,
) -> Vec<BitSet> {
    let num_queries = query_strings.len();
    let num_metadata = metadata_strings.len();

    let mut query_bitmaps: Vec<BitSet> = vec![BitSet::new(); num_queries];
    let mut query_labels: Vec<MultiLabel> = Vec::with_capacity(num_queries);
    let mut metadata_labels: Vec<MultiLabel> = Vec::with_capacity(num_metadata);

    // Parallel processing for query labels
    query_strings.iter().for_each(|label| {
        query_labels.push(MultiLabel::from_query(label));
    });

    // Parallel processing for metadata labels
    metadata_strings.iter().for_each(|label| {
        metadata_labels.push(MultiLabel::from_base(label));
    });

    // Prepare query_bitmaps
    let start_time = Instant::now();
    query_bitmaps
        .par_iter_mut()
        .enumerate()
        .for_each_in_pool(pool, |(query_idx, bitmap)| {
            for (metadata_idx, metadata_label) in metadata_labels.iter().enumerate() {
                if query_labels[query_idx].is_subset_of(metadata_label) {
                    bitmap.insert(metadata_idx);
                }
            }
        });

    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);

    query_bitmaps
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use diskann_providers::utils::create_thread_pool;

    use super::*;
    static POOL: LazyLock<RayonThreadPool> = LazyLock::new(|| create_thread_pool(4).unwrap());

    #[test]
    fn test_process_bitmap_for_labels() {
        let query_strings = vec![
            String::from("CAT=ExteriorAccessories&RATING=4|RATING=5"),
            String::from("CAT=ExteriorAccessories&RATING=5"),
            String::from(
                "CAT=Automotive&RATING=4|RATING=5&CAT=ReplacementParts|CAT=ExteriorAccessories",
            ),
            String::from("CAT=ReplacementParts&RATING=5"),
        ];

        let metadata_strings = vec![
            String::from("BRAND=Caltric,CAT=Automotive,CAT=ReplacementParts,CAT=MotorcyclePowersports,CAT=Parts,CAT=Filters,CAT=OilFilters,RATING=5"),
            String::from("BRAND=APL,CAT=Automotive,CAT=TiresWheels,CAT=AccessoriesParts,CAT=LugNutsAccessories,CAT=LugNuts,RATING=4"),
            String::from("BRAND=Cardone,CAT=Automotive,CAT=ReplacementParts,CAT=BrakeSystem,CAT=CalipersParts,CAT=CaliperBrackets,RATING=5"),
            String::from("BRAND=Monroe,CAT=Automotive,CAT=ReplacementParts,CAT=ShocksStrutsSuspension,CAT=Stabilizers,RATING=5"),
            String::from("BRAND=SEGADEN,CAT=Automotive,CAT=ExteriorAccessories,RATING=4"),
        ];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 4);

        //assertions based on expected behavior
        assert!(bitmaps[0].contains(4));
        assert!(!bitmaps[1].contains(4));
        assert!(bitmaps[2].contains(0));
        assert!(bitmaps[2].contains(2));
        assert!(bitmaps[2].contains(3));
        assert!(bitmaps[2].contains(4));
        assert!(bitmaps[3].contains(2));
        assert!(bitmaps[3].contains(3));
    }

    #[test]
    fn test_empty_query_strings() {
        let query_strings = vec![];
        let metadata_strings = vec![
            String::from("BRAND=Caltric,CAT=Automotive,CAT=MotorcyclePowersports,CAT=Parts,CAT=Filters,CAT=OilFilters,RATING=5"),
        ];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 0);
    }

    #[test]
    fn test_empty_metadata_strings() {
        let query_strings = vec![String::from("CAT=ExteriorAccessories&RATING=4|RATING=5")];
        let metadata_strings = vec![];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 1);
        assert!(bitmaps[0].is_empty());
    }

    #[test]
    fn test_serializable_bitset_conversion() {
        let mut bitset = BitSet::new();
        bitset.insert(0);
        bitset.insert(5);
        bitset.insert(10);

        let serializable = SerializableBitSet::from(&bitset);
        let converted_back: BitSet = serializable.into();

        assert!(converted_back.contains(0));
        assert!(converted_back.contains(5));
        assert!(converted_back.contains(10));
        assert!(!converted_back.contains(1));
    }

    #[test]
    fn test_serializable_bitset_empty() {
        let bitset = BitSet::new();
        let serializable = SerializableBitSet::from(&bitset);
        let converted_back: BitSet = serializable.into();
        assert!(converted_back.is_empty());
    }

    #[test]
    fn test_process_bitmap_single_query_single_metadata() {
        let query_strings = vec![String::from("CAT=Automotive")];
        let metadata_strings = vec![String::from("CAT=Automotive,RATING=5")];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 1);
        assert!(bitmaps[0].contains(0));
    }

    #[test]
    fn test_process_bitmap_no_match() {
        let query_strings = vec![String::from("CAT=Electronics")];
        let metadata_strings = vec![
            String::from("CAT=Automotive,RATING=5"),
            String::from("CAT=Fashion,RATING=4"),
        ];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 1);
        assert!(bitmaps[0].is_empty());
    }

    #[test]
    fn test_process_bitmap_multiple_matches() {
        let query_strings = vec![String::from("RATING=5")];
        let metadata_strings = vec![
            String::from("CAT=Automotive,RATING=5"),
            String::from("CAT=Fashion,RATING=4"),
            String::from("CAT=Electronics,RATING=5"),
        ];

        let bitmaps = process_bitmap_for_labels(query_strings, metadata_strings, &POOL);
        assert_eq!(bitmaps.len(), 1);
        assert!(bitmaps[0].contains(0));
        assert!(!bitmaps[0].contains(1));
        assert!(bitmaps[0].contains(2));
    }
}
