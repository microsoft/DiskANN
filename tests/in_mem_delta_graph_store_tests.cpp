// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

#include "in_mem_static_graph_store.h"
#include "in_mem_static_graph_reformat_store.h"

namespace
{
class ScopedGraphFile
{
  public:
    explicit ScopedGraphFile(const std::string& name)
        : path((std::filesystem::temp_directory_path() / name).string())
    {
    }

    ~ScopedGraphFile()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    std::string path;
};

void write_legacy_graph(
    const std::string& path,
    const std::vector<std::vector<uint32_t>>& graph,
    uint32_t start)
{
    const size_t metadata_size =
        sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
    size_t file_size = metadata_size;
    uint32_t max_degree = 0;
    for (const auto& neighbors : graph)
    {
        file_size += sizeof(uint32_t) * (neighbors.size() + 1);
        max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const size_t frozen_points = 1;
    out.write(reinterpret_cast<const char*>(&file_size), sizeof(file_size));
    out.write(reinterpret_cast<const char*>(&max_degree), sizeof(max_degree));
    out.write(reinterpret_cast<const char*>(&start), sizeof(start));
    out.write(reinterpret_cast<const char*>(&frozen_points), sizeof(frozen_points));
    for (const auto& neighbors : graph)
    {
        const uint32_t degree = static_cast<uint32_t>(neighbors.size());
        out.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
        out.write(
            reinterpret_cast<const char*>(neighbors.data()),
            static_cast<std::streamsize>(neighbors.size() * sizeof(uint32_t)));
    }
}

void write_reformatted_graph(
    const std::string& path,
    const std::vector<std::vector<uint32_t>>& graph)
{
    std::vector<size_t> offsets{0};
    uint32_t max_degree = 0;
    for (const auto& neighbors : graph)
    {
        offsets.push_back(offsets.back() + neighbors.size());
        max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
    }
    const size_t nodes = graph.size();
    const size_t frozen = 1;
    const uint32_t start = 0;
    const size_t file_size = 3 * sizeof(size_t) + 2 * sizeof(uint32_t)
        + offsets.size() * sizeof(size_t) + offsets.back() * sizeof(uint32_t);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char*>(&file_size), sizeof(file_size));
    out.write(reinterpret_cast<const char*>(&max_degree), sizeof(max_degree));
    out.write(reinterpret_cast<const char*>(&start), sizeof(start));
    out.write(reinterpret_cast<const char*>(&frozen), sizeof(frozen));
    out.write(reinterpret_cast<const char*>(&nodes), sizeof(nodes));
    out.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(size_t));
    for (const auto& neighbors : graph)
    {
        if (!neighbors.empty())
            out.write(reinterpret_cast<const char*>(neighbors.data()), neighbors.size() * sizeof(uint32_t));
    }
}

template <typename Store>
class InspectableGraphStore : public Store
{
  public:
    using Store::Store;

    bool has_compressed_storage() const
    {
        return std::holds_alternative<typename Store::StreamVByteGraphStorage>(this->_storage);
    }

    bool has_32_bit_offsets() const
    {
        const auto& compressed = std::get<typename Store::StreamVByteGraphStorage>(this->_storage);
        return std::holds_alternative<std::vector<uint32_t>>(compressed._offsets);
    }

    void use_64_bit_offsets()
    {
        auto& compressed = std::get<typename Store::StreamVByteGraphStorage>(this->_storage);
        const auto& offsets = std::get<std::vector<uint32_t>>(compressed._offsets);
        compressed._offsets = std::vector<uint64_t>(offsets.begin(), offsets.end());
    }
};

template <typename Store>
void check_storage(const std::string& path, const std::vector<std::vector<uint32_t>>& graph)
{
    for (bool compress : {false, true})
    {
        InspectableGraphStore<Store> store(graph.size(), 2, compress);
        for (int load = 0; load < 2; ++load)
        {
            store.load(path, graph.size());
            BOOST_TEST(store.has_compressed_storage() == compress);
            if (compress)
                BOOST_TEST(store.has_32_bit_offsets());

            auto check_neighbors = [&] {
                for (size_t node = 0; node < graph.size(); ++node)
                {
                    auto expected = graph[node];
                    if (compress)
                        std::sort(expected.begin(), expected.end());
                    std::vector<uint32_t> decoded;
                    store.get_neighbours(static_cast<uint32_t>(node)).convert_to_vector(decoded);
                    BOOST_TEST(decoded == expected, boost::test_tools::per_element());
                }
            };
            check_neighbors();
            if (compress)
            {
                // Exercise the wide-offset reader without allocating a 4 GiB payload.
                store.use_64_bit_offsets();
                BOOST_TEST(!store.has_32_bit_offsets());
                check_neighbors();
            }
        }
    }
}
} // namespace

BOOST_AUTO_TEST_SUITE(InMemDeltaGraphStore_tests)

BOOST_AUTO_TEST_CASE(loads_legacy_graph_into_delta_storage)
{
    const std::vector<std::vector<uint32_t>> graph = {
        {3, 1},
        {2},
        {},
        {0}
    };
    ScopedGraphFile file("diskann_delta_legacy_graph.bin");
    write_legacy_graph(file.path, graph, 0);

    diskann::InMemStaticGraphStore store(4, 2, true);
    const auto [nodes, start, frozen] = store.load(file.path, graph.size());

    BOOST_TEST(nodes == 4);
    BOOST_TEST(start == 0);
    BOOST_TEST(frozen == 1);

    const auto node0 = store.get_neighbours(0);
    std::vector<uint32_t> decoded0;
    node0.convert_to_vector(decoded0);
    BOOST_TEST(decoded0 == std::vector<uint32_t>({1, 3}), boost::test_tools::per_element());

    const auto node2 = store.get_neighbours(2);
    BOOST_TEST(node2.empty());

    BOOST_TEST(store.get_graph_size() > 0);
}

BOOST_AUTO_TEST_CASE(storage_variants_support_both_file_formats_and_reload)
{
    const std::vector<std::vector<uint32_t>> graph = {{3, 1}, {2}, {}, {0}};
    ScopedGraphFile legacy("diskann_variant_legacy_graph.bin");
    ScopedGraphFile reformatted("diskann_variant_reformatted_graph.bin");
    write_legacy_graph(legacy.path, graph, 0);
    write_reformatted_graph(reformatted.path, graph);
    check_storage<diskann::InMemStaticGraphStore>(legacy.path, graph);
    check_storage<diskann::InMemStaticGraphReformatStore>(reformatted.path, graph);
}

BOOST_AUTO_TEST_SUITE_END()
