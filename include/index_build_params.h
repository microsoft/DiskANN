#include "common_includes.h"
#include "parameters.h"

namespace diskann
{
struct IndexBuildParams
{
  public:
    diskann::IndexWriteParameters index_write_params;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;

  private:
    IndexBuildParams(IndexWriteParameters &index_write_params, std::string &save_path_prefix, std::string &label_file,
                     std::string &universal_label, uint32_t filter_threshold)
        : index_write_params(index_write_params), save_path_prefix(save_path_prefix), label_file(label_file),
          universal_label(universal_label), filter_threshold(filter_threshold)
    {
    }

    friend class IndexBuildParamsBuilder;
};
class IndexBuildParamsBuilder
{
  public:
    IndexBuildParamsBuilder(diskann::IndexWriteParameters &paras) : index_write_params(paras){};

    IndexBuildParamsBuilder &with_save_path_prefix(std::string &save_path_prefix)
    {
        if (save_path_prefix.empty() || save_path_prefix == "")
            throw ANNException("Error: save_path_prefix can't be empty", -1);
        this->save_path_prefix = save_path_prefix;
        return *this;
    }

    IndexBuildParamsBuilder &with_label_file(std::string &label_file)
    {
        this->label_file = label_file;
        return *this;
    }

    IndexBuildParamsBuilder &with_universal_label(std::string &univeral_label)
    {
        this->universal_label = univeral_label;
        return *this;
    }

    IndexBuildParamsBuilder &with_filter_threshold(std::uint32_t &filter_threshold)
    {
        this->filter_threshold = filter_threshold;
        return *this;
    }

    IndexBuildParams build()
    {
        return IndexBuildParams(index_write_params, save_path_prefix, label_file, universal_label, filter_threshold);
    }

    IndexBuildParamsBuilder(const IndexBuildParamsBuilder &) = delete;
    IndexBuildParamsBuilder &operator=(const IndexBuildParamsBuilder &) = delete;

  private:
    diskann::IndexWriteParameters index_write_params;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;
};
} // namespace diskann