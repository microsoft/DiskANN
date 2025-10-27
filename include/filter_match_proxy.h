#pragma once
#include "label_bitmask.h"
#include "integer_label_vector.h"

namespace diskann
{

    class filter_match_proxy
    {
    public:
        virtual bool contain_filtered_label(uint32_t id) = 0;
    };

    template <typename LabelT>
    class bitmask_filter_match : public filter_match_proxy
    {
    public:
        bitmask_filter_match(simple_bitmask_buf& bitmask_filters,
            std::vector<std::uint64_t>& query_bitmask_buf,
            const std::vector<LabelT>& filter_labels,
            LabelT unv_label);

        virtual bool contain_filtered_label(uint32_t id) override;

    private:
        simple_bitmask_buf& _bitmask_filters;
        std::vector<std::uint64_t>& _query_bitmask_buf;
        simple_bitmask_full_val _bitmask_full_val;
    };

    template <typename LabelT>
    class integer_label_filter_match : public filter_match_proxy
    {
    public:
        integer_label_filter_match(integer_label_vector& label_vector,
            const std::vector<LabelT>& filter_labels,
            LabelT unv_label);

        virtual bool contain_filtered_label(uint32_t id) override;

    private:
        integer_label_vector& _label_vector;
        const std::vector<LabelT>& _filter_labels;
        LabelT _unv_label;
    };

template <typename LabelT>
class label_filter_match_holder : public filter_match_proxy
{
public:
    label_filter_match_holder(simple_bitmask_buf& bitmask_filters,
        std::vector<std::uint64_t>& query_bitmask_buf,
        integer_label_vector& label_vector,
        const std::vector<LabelT>& filter_labels,
        LabelT unv_label,
        bool use_integer_labels);

    virtual bool contain_filtered_label(uint32_t id) override;

private:
    bitmask_filter_match<LabelT> _bitmask_filter_match;
    integer_label_filter_match<LabelT> _integer_label_filter_match;
    bool _use_integer_labels;
};

}