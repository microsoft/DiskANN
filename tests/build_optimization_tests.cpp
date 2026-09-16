// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann_exception.h"
#include "in_mem_graph_store.h"
#include "index.h"
#include "parameters.h"
#include "utils.h"

using namespace diskann;

namespace
{
namespace fs = std::filesystem;

class TestDirectory
{
  public:
    TestDirectory()
    {
        static std::atomic<uint64_t> sequence{0};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        _path = "build_optimization_test_" + std::to_string(timestamp) + "_" +
                std::to_string(sequence.fetch_add(1));
        if (!fs::create_directory(_path))
        {
            throw std::runtime_error("Failed to create test directory: " + _path.string());
        }
    }

    ~TestDirectory()
    {
        std::error_code error;
        fs::remove_all(_path, error);
    }

    std::string path(const char *name) const
    {
        return (_path / fs::path(name)).string();
    }

  private:
    fs::path _path;
};

std::string read_file(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
    {
        throw std::runtime_error("Failed to open test file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string line_ending()
{
#ifdef _WIN32
    return "\r\n";
#else
    return "\n";
#endif
}

template <typename ValueT> ValueT read_value(std::ifstream &input)
{
    ValueT value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input)
    {
        throw std::runtime_error("Failed to read graph test fixture");
    }
    return value;
}

void write_float_bin(const std::string &path, uint32_t point_count, uint32_t dimension)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error("Failed to create data fixture");
    }

    const int32_t points = static_cast<int32_t>(point_count);
    const int32_t dimensions = static_cast<int32_t>(dimension);
    output.write(reinterpret_cast<const char *>(&points), sizeof(points));
    output.write(reinterpret_cast<const char *>(&dimensions), sizeof(dimensions));
    for (uint32_t point = 0; point < point_count; ++point)
    {
        for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate)
        {
            uint64_t bits = 42ULL * 1103515245ULL + point * 12345ULL +
                            coordinate * 7919ULL + 12345ULL;
            bits ^= bits >> 21;
            bits *= 2685821657736338717ULL;
            bits ^= bits >> 31;
            const float value =
                (static_cast<float>(static_cast<uint32_t>(bits)) / 2147483648.0f) - 1.0f;
            output.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
    }
}
} // namespace

BOOST_AUTO_TEST_CASE(ConvertLabelsPreservesLayoutAndMapping)
{
    TestDirectory directory;
    const std::string input_path = directory.path("labels_input.txt");
    const std::string output_path = directory.path("labels_output.txt");
    const std::string map_path = directory.path("labels_map.txt");

    {
        std::ofstream input(input_path, std::ios::binary);
        BOOST_REQUIRE(input.is_open());
        input << "red,blue\n"
                 "blue\n"
                 "green,red\n";
    }

    uint32_t universal_label_id = 0;
    convert_labels_string_to_int(input_path, output_path, map_path, "blue", universal_label_id);

    const std::string expected =
        "1,2" + line_ending() + "2" + line_ending() + "3,1" + line_ending();
    BOOST_REQUIRE_EQUAL(read_file(output_path), expected);
    BOOST_REQUIRE_EQUAL(universal_label_id, 2);

    std::unordered_map<std::string, uint32_t> labels;
    std::ifstream map_file(map_path);
    BOOST_REQUIRE(map_file.is_open());
    for (std::string line; std::getline(map_file, line);)
    {
        std::istringstream entry(line);
        std::string label;
        uint32_t id = 0;
        BOOST_REQUIRE(static_cast<bool>(std::getline(entry, label, '\t')));
        BOOST_REQUIRE(entry >> id);
        labels.emplace(std::move(label), id);
    }
    BOOST_REQUIRE_EQUAL(labels.at("red"), 1);
    BOOST_REQUIRE_EQUAL(labels.at("blue"), 2);
    BOOST_REQUIRE_EQUAL(labels.at("green"), 3);
}

BOOST_AUTO_TEST_CASE(ConvertLabelsReportsOpenFailures)
{
    TestDirectory directory;
    const std::string missing_path = directory.path("missing_labels.txt");
    const std::string input_path = directory.path("valid_labels.txt");
    const std::string output_path = directory.path("failure_output.txt");
    const std::string map_path = directory.path("failure_map.txt");

    uint32_t universal_label_id = 0;
    BOOST_REQUIRE_THROW(
        convert_labels_string_to_int(missing_path, output_path, map_path, "", universal_label_id),
        ANNException);

    {
        std::ofstream input(input_path);
        BOOST_REQUIRE(input.is_open());
        input << "label\n";
    }

    BOOST_REQUIRE_THROW(
        convert_labels_string_to_int(input_path, ".", map_path, "", universal_label_id),
        ANNException);
    BOOST_REQUIRE_THROW(
        convert_labels_string_to_int(input_path, output_path, ".", "", universal_label_id),
        ANNException);
}

BOOST_AUTO_TEST_CASE(CopyAndMoveFilePreserveExpectedOwnership)
{
    TestDirectory directory;
    const std::string copy_source = directory.path("copy_source.bin");
    const std::string copy_destination = directory.path("copy_destination.bin");
    const std::string move_source = directory.path("move_source.bin");
    const std::string move_destination = directory.path("move_destination.bin");

    const std::string binary_content("new\0bytes", 9);
    {
        std::ofstream source(copy_source, std::ios::binary);
        source.write(binary_content.data(), binary_content.size());
        std::ofstream destination(copy_destination, std::ios::binary);
        destination << "old-longer-content";
    }

    copy_file(copy_source, copy_destination);
    BOOST_REQUIRE(fs::exists(copy_source));
    BOOST_REQUIRE_EQUAL(read_file(copy_destination), binary_content);

    {
        std::ofstream source(move_source, std::ios::binary);
        source << "replacement";
        std::ofstream destination(move_destination, std::ios::binary);
        destination << "old-longer-content";
    }

    move_file(move_source, move_destination);
    BOOST_REQUIRE(!fs::exists(move_source));
    BOOST_REQUIRE_EQUAL(read_file(move_destination), "replacement");
}

BOOST_AUTO_TEST_CASE(InMemoryGraphStorePreservesLegacyBinaryLayout)
{
    TestDirectory directory;
    const std::string graph_path = directory.path("graph.bin");

    InMemGraphStore graph(4, 3);
    std::vector<uint32_t> neighbors0{1, 2};
    std::vector<uint32_t> neighbors1;
    std::vector<uint32_t> neighbors2{0, 1, 3};
    std::vector<uint32_t> neighbors3{2};
    graph.set_neighbours(0, neighbors0);
    graph.set_neighbours(1, neighbors1);
    graph.set_neighbours(2, neighbors2);
    graph.set_neighbours(3, neighbors3);

    constexpr uint64_t expected_size = 64;
    BOOST_REQUIRE_EQUAL(graph.store(graph_path, 4, 99, 2), expected_size);
    BOOST_REQUIRE_EQUAL(fs::file_size(graph_path), expected_size);

    std::ifstream input(graph_path, std::ios::binary);
    BOOST_REQUIRE(input.is_open());
    BOOST_REQUIRE_EQUAL(read_value<uint64_t>(input), expected_size);
    BOOST_REQUIRE_EQUAL(read_value<uint32_t>(input), 3);
    BOOST_REQUIRE_EQUAL(read_value<uint32_t>(input), 2);
    BOOST_REQUIRE_EQUAL(read_value<uint64_t>(input), 1);

    const std::vector<std::vector<uint32_t>> expected{
        neighbors0, neighbors1, neighbors2, neighbors3};
    for (const auto &neighbors : expected)
    {
        BOOST_REQUIRE_EQUAL(read_value<uint32_t>(input), neighbors.size());
        for (const uint32_t neighbor : neighbors)
        {
            BOOST_REQUIRE_EQUAL(read_value<uint32_t>(input), neighbor);
        }
    }
    BOOST_REQUIRE_EQUAL(input.peek(), std::char_traits<char>::eof());
}

BOOST_AUTO_TEST_CASE(FilteredIndexSavePreservesNumericLabelLayout)
{
    constexpr uint32_t point_count = 64;
    constexpr uint32_t dimension = 8;
    constexpr uint32_t max_degree = 8;
    constexpr uint32_t search_list_size = 16;

    TestDirectory directory;
    const std::string data_path = directory.path("index_data.bin");
    const std::string raw_labels_path = directory.path("index_raw_labels.txt");
    const std::string index_prefix = directory.path("filtered_index");

    write_float_bin(data_path, point_count, dimension);

    std::string expected_labels;
    {
        std::ofstream labels(raw_labels_path, std::ios::binary);
        BOOST_REQUIRE(labels.is_open());
        for (uint32_t point = 0; point < point_count; ++point)
        {
            if (point == 0)
            {
                labels << "red,blue\n";
                expected_labels += "1,2" + line_ending();
            }
            else if ((point & 1U) == 0)
            {
                labels << "red\n";
                expected_labels += "1" + line_ending();
            }
            else
            {
                labels << "blue\n";
                expected_labels += "2" + line_ending();
            }
        }
    }

    auto write_parameters = std::make_shared<IndexWriteParameters>(
        IndexWriteParametersBuilder(search_list_size, max_degree)
            .with_alpha(1.2f)
            .with_num_threads(1)
            .with_filter_list_size(search_list_size)
            .build());
    Index<float, uint32_t, uint32_t> index(
        Metric::L2,
        dimension,
        point_count,
        write_parameters,
        nullptr,
        0,
        false,
        false,
        false,
        false,
        0,
        false,
        true);

    IndexFilterParams filters = IndexFilterParamsBuilder()
                                    .with_label_file(raw_labels_path)
                                    .with_save_path_prefix(index_prefix)
                                    .build();
    index.build(data_path, point_count, filters);
    index.save(index_prefix.c_str());

    BOOST_REQUIRE_EQUAL(read_file(index_prefix + "_labels.txt"), expected_labels);
}
