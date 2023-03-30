// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "defaults.h"

namespace diskann
{

class IndexWriteParameters

{
  public:
    const uint32_t search_list_size; // L
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_rounds;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf
    const uint32_t num_frozen_points;

  private:
    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_rounds,
                         const uint32_t num_threads, const uint32_t filter_list_size, const uint32_t num_frozen_points)
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_rounds(num_rounds), num_threads(num_threads),
          filter_list_size(filter_list_size), num_frozen_points(num_frozen_points)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t max_degree        // R
                                )
        : _search_list_size(search_list_size), _max_degree(max_degree)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_rounds(const uint32_t num_rounds)
    {
        _num_rounds = num_rounds;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_frozen_points(const uint32_t num_frozen_points)
    {
        _num_frozen_points = num_frozen_points;
        return *this;
    }

    IndexWriteParameters build() const
    {
        return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_rounds, _num_threads, _filter_list_size, _num_frozen_points);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _num_rounds(wp.num_rounds), _filter_list_size(wp.filter_list_size), _num_frozen_points(wp.num_frozen_points)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_rounds{defaults::NUM_ROUNDS};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
    uint32_t _num_frozen_points{defaults::NUM_FROZEN_POINTS};
};

class IndexReadParameters
{
    /**
     * Parameters used for reading/searching the index.
     * Simple constructor and properties because they're < 4 properties
     *
     */
  public:
    const uint32_t search_list_size;
    const uint32_t num_threads;
    IndexReadParameters(const uint32_t search_list_size, const uint32_t num_threads)
        : search_list_size(search_list_size), num_threads(num_threads)
    {
    }

    IndexReadParameters &operator=(const IndexReadParameters &) = delete;
};

class Parameters
{
  public:
    Parameters()
    {
        int *p = new int;
        *p = 0;
        params["num_threads"] = p;
    }

    template <typename ParamType> inline void Set(const std::string &name, const ParamType &value)
    {
        //      ParamType *ptr = (ParamType *) malloc(sizeof(ParamType));
        if (params.find(name) != params.end())
        {
            free(params[name]);
        }
        ParamType *ptr = new ParamType;
        *ptr = value;
        params[name] = (void *)ptr;
    }

    template <typename ParamType> inline ParamType Get(const std::string &name) const
    {
        auto item = params.find(name);
        if (item == params.end())
        {
            throw std::invalid_argument("Invalid parameter name.");
        }
        else
        {
            // return ConvertStrToValue<ParamType>(item->second);
            if (item->second == nullptr)
            {
                throw std::invalid_argument(std::string("Parameter ") + name + " has value null.");
            }
            else
            {
                return *(static_cast<ParamType *>(item->second));
            }
        }
    }

    template <typename ParamType> inline ParamType Get(const std::string &name, const ParamType &default_value) const
    {
        try
        {
            return Get<ParamType>(name);
        }
        catch (std::invalid_argument e)
        {
            return default_value;
        }
    }

    ~Parameters()
    {
        for (auto iter = params.begin(); iter != params.end(); iter++)
        {
            if (iter->second != nullptr)
                free(iter->second);
            // delete iter->second;
        }
    }

  private:
    std::unordered_map<std::string, void *> params;

    Parameters(const Parameters &);
    Parameters &operator=(const Parameters &);

    template <typename ParamType> inline ParamType ConvertStrToValue(const std::string &str) const
    {
        std::stringstream sstream(str);
        ParamType value;
        if (!(sstream >> value) || !sstream.eof())
        {
            std::stringstream err;
            err << "Failed to convert value '" << str << "' to type: " << typeid(value).name();
            throw std::runtime_error(err.str());
        }
        return value;
    }
};
} // namespace diskann
