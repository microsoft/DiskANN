use std::sync::Arc;

use diskann::{
    graph::{
        DiskANNIndex,
        test::{provider, synthetic},
    },
    utils::{IntoUsize, ONE},
};

use crate::build::{self, graph::SingleInsert};

/// Construct a test index over a 4-dimensional grid with edge-size 4.
///
/// Returns the constructed index as well as the number of non-start points in the index.
pub(super) fn build_test_index() -> (Arc<DiskANNIndex<provider::Provider>>, usize) {
    let grid = synthetic::Grid::Four;
    let size = 4;
    let start_id = u32::MAX;
    let distance = diskann_vector::distance::Metric::L2;

    let start_point = grid.start_point(size);
    let data = Arc::new(grid.data(size));

    let provider_config = provider::Config::new(
        distance,
        2 * grid.dim().into_usize(),
        std::iter::once(provider::StartPoint::new(start_id, start_point)),
    )
    .unwrap();

    let provider = provider::Provider::new(provider_config);

    let index_config = diskann::graph::config::Builder::new(
        provider.max_degree().checked_sub(3).unwrap(),
        diskann::graph::config::MaxDegree::new(provider.max_degree()),
        20,
        distance.into(),
    )
    .build()
    .unwrap();

    let index = Arc::new(diskann::graph::DiskANNIndex::new(
        index_config,
        provider,
        None,
    ));

    let rt = crate::tokio::runtime(1).unwrap();
    let _ = build::build(
        SingleInsert::new(
            index.clone(),
            data.clone(),
            provider::Strategy::new(),
            build::ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(ONE, ONE),
        &rt,
    )
    .unwrap();

    (index, data.nrows())
}
