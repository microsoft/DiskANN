// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include <memory>
#include "in_mem_data_store.h"

#include "filter_utils.h"

namespace diskann
{

template <typename label_t>
InMemFilterStore<label_t>::InMemFilterStore(const location_t num_points)
    : AbstractFilterStore<label_t>(num_points)
{
    _pts_to_labels.resize(num_points);
    _universal_label_exists = false;
}

template <typename label_t> InMemFilterStore<data_t>::~InMemFilterStore()
{
    delete _pts_to_labels;
}

template <typename label_t> location_t load(const std::string &filename) 
{
    std::string labels_map_file = filename + "_labels_map.txt";
    _labels_file = filename + "_labels.txt";
    load_label_map(labels_map_file);
    // load_labels from _label_file & include parse_label_file logic
    location_t num_points = load_labels();
    load_medoids(filename);
    load_universal_label(filename);
    return num_points;
}

template <typename label_t> location_t load_labels()
{
    std::ifstream infile(_label_file);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    std::string line, token;
    uint32_t line_cnt = 0;
    line_cnt = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<label_t> lbls(0);
        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            label_t token_as_num = (label_t)std::stoul(token);
            lbls.push_back(token_as_num);
            _labels_set.insert(token_as_num);
        }
        if (lbls.size() <= 0)
        {
            diskann::cout << "No label found";
            exit(-1);
        }
        std::sort(lbls.begin(), lbls.end());
        _pts_to_labels[line_cnt] = lbls;
        line_cnt++;
    }
    assert(line_cnt == this->get_number_points());
    diskann::cout << "Identified " << _labels_set.size() << " distinct label(s)" << std::endl;

    return (location_t)line_cnt;
}

template <typename label_t> size_t save(const std::string &filename, const location_t num_pts)
{
    save_labels(filename);
    save_medoids(filename);
    save_universal_label(filename);
}

template <typename label_t> void set_universal_label(const label_t label)
{
    _universal_label = label;
}

template <typename label_t> void set_labels(const location_t i, std::vector<label_t> labels)
{
    _pts_to_labels[i] = labels;
    return;
}

template <typename label_t> location_t get_medoid(const label_t label) const
{
    return _label_to_medoid_id[label]
}

template <typename label_t> label_t get_universal_label() const
{
    return _universal_label;
}

template <typename label_t> std::vector<label_t> get_labels_by_point(const location_t point_id) const
{
    return _pts_to_labels[point_id];
}

template <typename label_t> label_t get_label(const std::string& raw_label) const
{
    if (_label_map.find(raw_label) != _label_map.end())
    {
        return _label_map[raw_label];
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename label_t> location_t calculate_medoids()
{
    std::unordered_map<LabelT, std::vector<uint32_t>> label_to_points;

    for (uint32_t point_id = 0; point_id < num_points_to_load; point_id++)
    {
        for (auto label : _pts_to_labels[point_id])
        {
            if (label != _universal_label)
            {
                label_to_points[label].emplace_back(point_id);
            }
            else
            {
                for (typename tsl::robin_set<LabelT>::size_type lbl = 0; lbl < _labels.size(); lbl++)
                {
                    auto itr = _labels.begin();
                    std::advance(itr, lbl);
                    auto &x = *itr;
                    label_to_points[x].emplace_back(point_id);
                }
            }
        }
    }

    uint32_t num_cands = 25;
    for (auto itr = _labels.begin(); itr != _labels.end(); itr++)
    {
        uint32_t best_medoid_count = std::numeric_limits<uint32_t>::max();
        auto &curr_label = *itr;
        uint32_t best_medoid;
        auto labeled_points = label_to_points[curr_label];
        for (uint32_t cnd = 0; cnd < num_cands; cnd++)
        {
            uint32_t cur_cnd = labeled_points[rand() % labeled_points.size()];
            uint32_t cur_cnt = std::numeric_limits<uint32_t>::max();
            if (_medoid_counts.find(cur_cnd) == _medoid_counts.end())
            {
                _medoid_counts[cur_cnd] = 0;
                cur_cnt = 0;
            }
            else
            {
                cur_cnt = _medoid_counts[cur_cnd];
            }
            if (cur_cnt < best_medoid_count)
            {
                best_medoid_count = cur_cnt;
                best_medoid = cur_cnd;
            }
        }
        _label_to_medoid_id[curr_label] = best_medoid;
        _medoid_counts[best_medoid]++;
    }
}

template <typename label_t> void prepare_label_file(const std::string &filename, const std::string& raw_universal_label, const std::string& save_path_prefix)
{
    _label_file = save_path_prefix +"_prepared_labels.txt";
    std::ofstream label_writer(_label_file);
    std::ifstream label_reader(filename);
    std::string line, token;
    while (std::getline(label_reader, line))
    {
        std::istringstream new_iss(line);
        std::vector<uint32_t> lbls;
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            if (_labels_map.find(token) == _labels_map.end())
            {
                uint32_t nextId = (uint32_t)_labels_map.size() + 1;
                _labels_map[token] = nextId;
            }
            lbls.push_back(string_int_map[token]);
        }
        if (lbls.size() <= 0)
        {
            std::cout << "No label found";
            exit(-1);
        }
        for (size_t j = 0; j < lbls.size(); j++)
        {
            if (j != lbls.size() - 1)
                label_writer << lbls[j] << ",";
            else
                label_writer << lbls[j] << std::endl;
        }
    }
    label_writer.close();
    save_raw_label_map(save_path_prefix);
}

template <typename label_t> bool detect_common_filters(location_t point_id, bool search_invocation, 
                          const std::vector<label_t> &incoming_labels) const
{
    auto &curr_node_labels = _pts_to_labels[point_id];
    std::vector<label_t> common_filters;
    std::set_intersection(incoming_labels.begin(), incoming_labels.end(), curr_node_labels.begin(),
                          curr_node_labels.end(), std::back_inserter(common_filters));
    if (common_filters.size() > 0)
    {
        // This is to reduce the repetitive calls. If common_filters size is > 0 ,
        // we dont need to check further for universal label
        return true;
    }
    if (_universal_label_exists)
    {
        if (!search_invocation)
        {
            if (std::find(incoming_labels.begin(), incoming_labels.end(), _universal_label) != incoming_labels.end() ||
                std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.push_back(_universal_label);
        }
        else
        {
            if (std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.push_back(_universal_label);
        }
    }
    return (common_filters.size() > 0);
}

template <typename label_t> std::vector<location_t> get_start_nodes() const
{
    std::vector<location_t> filter_specific_start_nodes;
    for (auto &x : _pts_to_labels[location])
    {
        filter_specific_start_nodes.emplace_back(_label_to_medoid_id[x]);
    }

    return filter_specific_start_nodes;
}

template <typename label_t> void save_labels(const std::string &filename)
{
    if (_pts_to_labels.size() > 0)
    {
        std::ofstream label_writer(std::string(filename) + "_labels.txt");
        assert(label_writer.is_open());
        for (uint32_t i = 0; i < _pts_to_labels.size(); i++)
        {
            for (uint32_t j = 0; j < (_pts_to_labels[i].size() - 1); j++)
            {
                label_writer << _pts_to_labels[i][j] << ",";
            }
            if (_pts_to_labels[i].size() != 0)
                label_writer << _pts_to_labels[i][_pts_to_labels[i].size() - 1];
            label_writer << std::endl;
        }
        label_writer.close();
    }
}

template <typename label_t> void save_medoids(const std::string &filename)
{
    if (_label_to_medoid_id.size() > 0)
    {
        std::ofstream medoid_writer(std::string(filename) + "_labels_to_medoids.txt");
        if (medoid_writer.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + filename, -1);
        }
        for (auto iter : _label_to_medoid_id)
        {
            medoid_writer << iter.first << ", " << iter.second << std::endl;
        }
        medoid_writer.close();
    }
}

template <typename label_t> void save_universal_label(const std::string &filename)
{
    if (_universal_label_exists)
    {
        std::ofstream universal_label_writer(std::string(filename) + "_universal_label.txt");
        assert(universal_label_writer.is_open());
        universal_label_writer << _universal_label << std::endl;
        universal_label_writer.close();
    }

}

template <typename label_t> void save_raw_label_map(const std::string &save_path_prefix)
{
    std::string mapFileName = save_path_prefix + "_labels_map.txt";
    std::ofstream map_writer(mapFileName);
    for (auto mp : _labels_map)
    {
        map_writer << mp.first << "\t" << mp.second << std::endl;
    }
    map_writer.close();
}


template <typename label_t> void load_medoids(const std::string &filename)
{
    if (file_exists(filename))
    {
        std::ifstream medoid_stream(filename);
        std::string line, token;
        uint32_t line_cnt = 0;

        _label_to_medoid_id.clear();

        while (std::getline(medoid_stream, line))
        {
            std::istringstream iss(line);
            uint32_t cnt = 0;
            uint32_t medoid = 0;
            LabelT label;
            while (std::getline(iss, token, ','))
            {
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                LabelT token_as_num = (LabelT)std::stoul(token);
                if (cnt == 0)
                    label = token_as_num;
                else
                    medoid = token_as_num;
                cnt++;
            }
            _label_to_medoid_id[label] = medoid;
            line_cnt++;
        }
    }
}

template <typename label_t> void load_universal_label (const std::string &filename)
{
    std::string universal_label_file(filename);
    universal_label_file += "_universal_label.txt";
    if (file_exists(universal_label_file))
    {
        std::ifstream universal_label_reader(universal_label_file);
        universal_label_reader >> _universal_label;
        _universal_label_exists = true;
        universal_label_reader.close();
    }
}

template <typename label_t> void load_raw_label_map(const std::string &filename)
{
    std::ifstream map_reader(labels_map_file);
    std::string line, token;
    labe_t label;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        label = (label_t)std::stoul(token);
        _label_map[label_str] = label;
    }
}

}// namespace diskann