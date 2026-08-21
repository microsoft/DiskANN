#include "diskann_memory_ffi.h"

#include <cstdint>
#include <type_traits>

static_assert(static_cast<int>(DiskANN::DiskANNError::SearchFailed) == 6);
static_assert(std::is_same_v<decltype(DiskANN::IndexConfiguration::dist_metric), DiskANN::Metric>);
static_assert(sizeof(DiskANN::IndexConfiguration) == 104);
static_assert(offsetof(DiskANN::IndexConfiguration, dist_metric) == 0);
static_assert(offsetof(DiskANN::IndexConfiguration, dim) == 8);
static_assert(offsetof(DiskANN::IndexConfiguration, search_list_size) == 16);
static_assert(offsetof(DiskANN::IndexConfiguration, num_threads) == 20);
static_assert(offsetof(DiskANN::IndexConfiguration, index_path) == 24);
static_assert(offsetof(DiskANN::IndexConfiguration, is_streaming) == 80);
static_assert(offsetof(DiskANN::IndexConfiguration, delete_method) == 84);
static_assert(offsetof(DiskANN::IndexConfiguration, delete_num_to_replace) == 88);
static_assert(offsetof(DiskANN::IndexConfiguration, delete_search_k) == 92);
static_assert(offsetof(DiskANN::IndexConfiguration, delete_search_l) == 96);
static_assert(sizeof(DiskANN::SearchParams) == 12);
static_assert(offsetof(DiskANN::SearchParams, beam_width) == 8);
static_assert(offsetof(DiskANN::SearchResult, indices) == 0);
static_assert(offsetof(DiskANN::SearchResult, distances) == 8);
static_assert(offsetof(DiskANN::SearchResult, result_count) == 16);
static_assert(sizeof(DiskANN::SearchResult) == 24);
static_assert(std::is_same_v<
              decltype(&DiskANN::diskann_load_memory_index_u8),
              DiskANN::DiskANNResult (*)(DiskANN::IndexConfiguration)>);
int main() {
  DiskANN::IndexConfiguration config{
      DiskANN::Metric::L2,
      1,
      1,
      1,
      "missing-index",
      static_cast<uint32_t>(DiskANN::TagType::U32),
      0.0f,
      0,
      0,
      0,
      0.0f,
      0,
      nullptr,
      nullptr,
      0,
      static_cast<uint32_t>(DiskANN::DeleteMethod::OneHop),
      3,
      10,
      64,
  };
  const auto result = DiskANN::diskann_load_memory_index_u8(config);
  DiskANN::diskann_free_memory_index(result.handle);
  return result.error == DiskANN::DiskANNError::LoadFailed ? 0 : 1;
}
