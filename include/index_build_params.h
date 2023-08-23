#include "common_includes.h"
#include "parameters.h"

namespace diskann
{
struct IndexBuildParams
{
  public:
    std::string save_path_prefix;
    std::string label_file;
    std::string tags_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;

  private:
    IndexBuildParams(const std::string &save_path_prefix, const std::string &label_file, const std::string &tags_file,
                     const std::string &universal_label, uint32_t filter_threshold)
        : save_path_prefix(save_path_prefix), label_file(label_file), tags_file(tags_file),
          universal_label(universal_label), filter_threshold(filter_threshold)
    {
    }

    friend class IndexBuildParamsBuilder;
};
class IndexBuildParamsBuilder
{
  public:
    IndexBuildParamsBuilder() = default;
    IndexBuildParamsBuilder(const IndexBuildParamsBuilder &) = delete;
    IndexBuildParamsBuilder &operator=(const IndexBuildParamsBuilder &) = delete;

    IndexBuildParamsBuilder &with_save_path_prefix(const std::string &save_path_prefix)
    {
        if (save_path_prefix.empty() || save_path_prefix == "")
            throw ANNException("Error: save_path_prefix can't be empty", -1);
        this->_save_path_prefix = save_path_prefix;
        return *this;
    }

    IndexBuildParamsBuilder &with_label_file(const std::string &label_file)
    {
        this->_label_file = label_file;
        return *this;
    }

    IndexBuildParamsBuilder &with_universal_label(const std::string &univeral_label)
    {
        this->_universal_label = univeral_label;
        return *this;
    }

    IndexBuildParamsBuilder &with_filter_threshold(const std::uint32_t &filter_threshold)
    {
        this->_filter_threshold = filter_threshold;
        return *this;
    }

    IndexBuildParamsBuilder &with_tags_file(const std::string &tags_file)
    {
        this->_tags_file = tags_file;
        return *this;
    }

    IndexBuildParams build()
    {
        return IndexBuildParams(_save_path_prefix, _label_file, _tags_file, _universal_label, _filter_threshold);
    }

  private:
    std::string _save_path_prefix;
    std::string _label_file;
    std::string _tags_file;
    std::string _universal_label;
    uint32_t _filter_threshold = 0;
};
} // namespace diskann
