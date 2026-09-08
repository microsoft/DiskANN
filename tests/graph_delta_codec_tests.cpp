// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <stdexcept>

#include "graph_delta_codec.h"

BOOST_AUTO_TEST_SUITE(GraphDeltaCodec_tests)

BOOST_AUTO_TEST_CASE(stream_vbyte_round_trips_multiple_groups)
{
    const std::vector<uint32_t> neighbors = {
        0xFFFFFF, 0, 1, 300, 70000, 70001, 900000, 900100, 12000000, 42
    };
    diskann::NeighborList list(neighbors.data(), neighbors.size());
    std::vector<uint32_t> sorted;
    const auto encoding =
        diskann::prepare_stream_vbyte_node(list, 1u << 24, 3, 1, sorted);

    std::vector<uint8_t> encoded(encoding.encoded_size);
    diskann::encode_stream_vbyte_node(
        sorted, encoding, 3, 1, encoded.data(), encoded.size());
    std::vector<uint32_t> decoded(neighbors.size());
    const size_t degree = diskann::decode_stream_vbyte_node(
        encoded.data(), encoded.size(), 1u << 24, 3, 1, decoded.data(), decoded.size());

    BOOST_TEST(degree == neighbors.size());
    BOOST_TEST(decoded == sorted, boost::test_tools::per_element());
    BOOST_TEST(encoding.control_bytes == 3);
}

BOOST_AUTO_TEST_CASE(stream_vbyte_round_trips_empty_and_singleton_nodes)
{
    for (const std::vector<uint32_t>& neighbors :
         {std::vector<uint32_t>{}, std::vector<uint32_t>{0xFFFFFF}})
    {
        diskann::NeighborList list(neighbors.data(), neighbors.size());
        std::vector<uint32_t> sorted;
        const auto encoding =
            diskann::prepare_stream_vbyte_node(list, 1u << 24, 3, 1, sorted);
        std::vector<uint8_t> encoded(encoding.encoded_size);
        diskann::encode_stream_vbyte_node(
            sorted, encoding, 3, 1, encoded.data(), encoded.size());
        std::vector<uint32_t> decoded(neighbors.size());
        const size_t degree = diskann::decode_stream_vbyte_node(
            encoded.data(), encoded.size(), 1u << 24, 3, 1, decoded.data(), decoded.size());
        BOOST_TEST(degree == neighbors.size());
        BOOST_TEST(decoded == sorted, boost::test_tools::per_element());
    }
}

BOOST_AUTO_TEST_CASE(stream_vbyte_rejects_truncated_payload)
{
    const std::vector<uint32_t> neighbors = {10, 20, 70000};
    diskann::NeighborList list(neighbors.data(), neighbors.size());
    std::vector<uint32_t> sorted;
    const auto encoding =
        diskann::prepare_stream_vbyte_node(list, 100000, 3, 1, sorted);
    std::vector<uint8_t> encoded(encoding.encoded_size);
    diskann::encode_stream_vbyte_node(
        sorted, encoding, 3, 1, encoded.data(), encoded.size());
    encoded.pop_back();

    std::vector<uint32_t> decoded(neighbors.size());
    BOOST_CHECK_THROW(
        diskann::decode_stream_vbyte_node(
            encoded.data(), encoded.size(), 100000, 3, 1, decoded.data(), decoded.size()),
        std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
