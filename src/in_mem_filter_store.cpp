#include <in_mem_filter_store.h>

namespace diskann
{

template <typename label_type>
InMemFilterStore<label_type>::InMemFilterStore(const size_t num_points) : AbstractFilterStore<label_type>(num_points)
{
    _location_to_labels.resize(num_points);
}

template <typename label_type>
bool InMemFilterStore<label_type>::detect_common_filters(uint32_t point_id, bool search_invocation,
                                                         const std::vector<label_type> &incoming_labels,
                                                         const FilterMatchStrategy filter_match_strategy)
{
    switch (filter_match_strategy)
    {
    case FilterMatchStrategy::SET_INTERSECTION:
        return this->detect_common_filters_by_set_intersection(point_id, search_invocation, incoming_labels);
    default:
        throw diskann::ANNException("Error: the provided filter match strategy is not supported.", -1);
    }
}

template <typename label_type>
const std::vector<label_type> &InMemFilterStore<label_type>::get_labels_by_location(const location_t point_id)
{
    return _location_to_labels[point_id];
}

template <typename label_type>
void InMemFilterStore<label_type>::set_labels_to_location(const location_t location,
                                                          const std::vector<std::string> &label_str)
{
    std::vector<label_type> labels;
    for (int i = 0; i < label_str.size(); i++)
    {
        labels.push_back(this->get_numeric_label(label_str[i]));
    }
    _location_to_labels[location] = labels;
}

template <typename label_type>
void InMemFilterStore<label_type>::swap_labels(const location_t location_first, const location_t location_second)
{
    _location_to_labels[location_first].swap(_location_to_labels[location_second]);
}

template <typename label_type> const tsl::robin_set<label_type> &InMemFilterStore<label_type>::get_all_label_set()
{
    return _labels;
}

template <typename label_type> void InMemFilterStore<label_type>::add_to_label_set(const label_type &label)
{
    _labels.insert(label);
}

template <typename label_type>
void InMemFilterStore<label_type>::add_label_to_location(const location_t point_id, const label_type label)
{
    _location_to_labels[point_id].emplace_back(label);
}

template <typename label_type>
void InMemFilterStore<label_type>::set_universal_label(const std::string &raw_universal_label)
{
    if (raw_universal_label.empty())
    {
        std::cout << "Warning: empty universal label passed" << std::endl;
    }
    else
    {
        _has_universal_label = true;
        _universal_label = _label_map[raw_universal_label];
    }
}

template <typename label_type> std::pair<bool, label_type> InMemFilterStore<label_type>::get_universal_label()
{
    std::pair<bool, label_type> universal_label;
    universal_label.second = _universal_label;
    if (_has_universal_label)
    {
        universal_label.first = false;
    }
    else
    {
        universal_label.second = false;
    }
    return universal_label;
}

// ideally takes raw label file and then genrate internal mapping and keep the info of mapping
template <typename label_type>
size_t InMemFilterStore<label_type>::populate_labels(const std::string &raw_labels_file,
                                                     const std::string &raw_universal_label)
{
    std::string raw_label_file_path =
        std::string(raw_labels_file).erase(raw_labels_file.size() - 4); // remove .txt from end
    // generate a map file
    std::string labels_file_to_use =
        raw_label_file_path + "_label_numeric.txt"; // will not be used after parse, can be safely deleted.
    std::string mem_labels_int_map_file = raw_label_file_path + "_labels_map.txt";
    _label_map = InMemFilterStore::convert_label_to_numeric(raw_labels_file, labels_file_to_use,
                                                            mem_labels_int_map_file, raw_universal_label);
    return load_labels(labels_file_to_use);
}

template <typename label_type> void InMemFilterStore<label_type>::load_label_map(const std::string &labels_map_file)
{
    if (file_exists(labels_map_file))
    {
        std::ifstream map_reader(labels_map_file);
        std::string line, token;
        label_type token_as_num;
        std::string label_str;
        while (std::getline(map_reader, line))
        {
            std::istringstream iss(line);
            getline(iss, token, '\t');
            label_str = token;
            getline(iss, token, '\t');
            token_as_num = (label_type)std::stoul(token);
            _label_map[label_str] = token_as_num;
        }
    }
    else
    {
        // TODO: throw exception from here and also make sure filtered_index is set appropriately for both build and
        // search of index.
        diskann::cout << "Warning: Can't load label map file please make sure it was generate, either by "
                         "filter_store->populate_labels() "
                         "then index->save() or  convert_label_to_numeric() method in case of dynamic index"
                      << std::endl;
    }
}

template <typename label_type>
void InMemFilterStore<label_type>::load_universal_labels(const std::string &universal_labels_file)
{
    if (file_exists(universal_labels_file))
    {
        std::ifstream universal_label_reader(universal_labels_file);
        std::string line;
        while (std::getline(universal_label_reader, line))
        {
            std::istringstream iss(line);
            label_type universal_label;
            if (!(iss >> universal_label))
            {
                throw std::runtime_error("ERROR: Invalid universal label " + line);
            }
            _universal_label = universal_label;
            _has_universal_label = true;
        }
        universal_label_reader.close();
    }
}

// load labels, labels_map and universal_label to filter store variables & returns total number of points
template <typename label_type> size_t InMemFilterStore<label_type>::load(const std::string &load_path)
{
    const std::string labels_file = load_path + "_labels.txt";
    const std::string labels_map_file = load_path + "_labels_map.txt";
    const std::string universal_label_file = load_path + "_universal_label.txt";
    load_label_map(labels_map_file);
    load_universal_labels(universal_label_file);
    return load_labels(labels_file);
}

template <typename label_type>
void InMemFilterStore<label_type>::save(const std::string &save_path, const size_t total_points)
{
    const std::string label_path = save_path + "_labels.txt";
    const std::string universal_label_path = save_path + "_universal_label.txt";
    const std::string label_map_path = save_path + "_labels_map.txt";
    save_label_map(label_map_path);
    save_universal_label(universal_label_path);
    save_labels(label_path, total_points);
}

template <typename label_type>
void InMemFilterStore<label_type>::save_labels(const std::string &save_path, const size_t total_points)
{

    if (_location_to_labels.size() > 0)
    {
        std::ofstream label_writer(save_path);
        assert(label_writer.is_open());
        for (uint32_t i = 0; i < total_points; i++)
        {
            for (uint32_t j = 0; j + 1 < _location_to_labels[i].size(); j++)
            {
                label_writer << _location_to_labels[i][j] << ",";
            }
            if (_location_to_labels[i].size() != 0)
                label_writer << _location_to_labels[i][_location_to_labels[i].size() - 1];
            label_writer << std::endl;
        }
        label_writer.close();
    }
}

template <typename label_type>
void InMemFilterStore<label_type>::save_raw_labels(const std::string &save_path, const size_t total_points)
{
    if (_label_map.empty())
    {
        diskann::cout << "Warning: not saving raw labels as label map is empty" << std::endl;
        return;
    }
    std::unordered_map<label_type, std::string> mapped_to_raw_labels;
    // invert label map
    for (const auto &[key, value] : _label_map)
    {
        mapped_to_raw_labels.insert({value, key});
    }

    // write updated labels
    std::ofstream raw_label_writer(save_path);
    assert(raw_label_writer.is_open());
    for (uint32_t i = 0; i < total_points; i++)
    {
        for (uint32_t j = 0; j + 1 < _location_to_labels[i].size(); j++)
        {
            raw_label_writer << mapped_to_raw_labels[_location_to_labels[i][j]] << ",";
        }
        if (_location_to_labels[i].size() != 0)
            raw_label_writer << mapped_to_raw_labels[_location_to_labels[i][_location_to_labels[i].size() - 1]];

        raw_label_writer << std::endl;
    }
    raw_label_writer.close();
}

template <typename label_type> void InMemFilterStore<label_type>::save_universal_label(const std::string &save_path)
{
    if (_has_universal_label)
    {
        std::ofstream universal_label_writer(save_path);
        assert(universal_label_writer.is_open());
        // universal_label_writer << _universal_label << std::endl;

        universal_label_writer << _universal_label << std::endl;
        universal_label_writer.close();
    }
}

template <typename label_type> void InMemFilterStore<label_type>::save_label_map(const std::string &save_path)
{
    if (_label_map.empty())
    {
        diskann::cout << "Warning: not saving label map as it is empty." << std::endl;
        return;
    }
    std::ofstream map_writer(save_path);
    for (auto mp : _label_map)
    {
        map_writer << mp.first << "\t" << mp.second << std::endl;
    }
    map_writer.close();
}

template <typename label_type> label_type InMemFilterStore<label_type>::get_numeric_label(const std::string &raw_label)
{
    if (_label_map.empty())
    {
        throw diskann::ANNException("Error: Label map is empty, please load the map before hand", -1);
    }
    if (_label_map.find(raw_label) != _label_map.end())
    {
        return _label_map[raw_label];
    }
    // why is this here
    if (_has_universal_label)
    {
        // Not sure why this is here, but when we start getting more labels chnage this
        return _universal_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename label_type> size_t InMemFilterStore<label_type>::load_labels(const std::string &label_file)
{
    // Format of Label txt file: filters with comma separators
    // Format of Label txt file: filters with comma separators
    std::ifstream infile(label_file, std::ios::binary);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    uint32_t line_cnt = 0;

    // Find total number of points in the labels file to reserve _locations_to_labels
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
            break;
        cur_pos = next_pos + 1;
        line_cnt++;
    }
    cur_pos = 0;
    next_pos = 0;
    _location_to_labels.resize(line_cnt, std::vector<label_type>());
    line_cnt = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }
        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        std::vector<label_type> lbls(0);
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }
            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }
            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }
            label_type token_as_num = (label_type)std::stoul(label_str);
            lbls.push_back(token_as_num);
            _labels.insert(token_as_num);
            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }
        cur_pos = next_pos + 1;
        _location_to_labels[line_cnt] = lbls;
        line_cnt++;
    }
    diskann::cout << "Identified " << _labels.size() << " distinct label(s)" << std::endl;
    return (size_t)line_cnt;
}

template <typename label_type>
std::unordered_map<std::string, label_type> InMemFilterStore<label_type>::convert_label_to_numeric(
    const std::string &inFileName, const std::string &outFileName, const std::string &mapFileName,
    const std::string &raw_universal_label)
{
    std::unordered_map<std::string, label_type> string_int_map;
    std::ofstream label_writer(outFileName);
    std::ifstream label_reader(inFileName);
    std::string line, token;
    if (raw_universal_label != "")
        string_int_map[raw_universal_label] = 0; // if universal label is provided map it to 0 always

    while (std::getline(label_reader, line))
    {
        std::istringstream new_iss(line);
        std::vector<uint32_t> lbls;
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            if (string_int_map.find(token) == string_int_map.end())
            {
                uint32_t nextId = (uint32_t)string_int_map.size();
                string_int_map[token] = nextId;
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

    std::ofstream map_writer(mapFileName);
    for (auto mp : string_int_map)
    {
        map_writer << mp.first << "\t" << mp.second << std::endl;
    }
    map_writer.close();

    return string_int_map;
}

template <typename label_type>
bool InMemFilterStore<label_type>::detect_common_filters_by_set_intersection(
    uint32_t point_id, bool search_invocation, const std::vector<label_type> &incoming_labels)
{
    auto &curr_node_labels = _location_to_labels[point_id];
    std::set<label_type> common_filters;
    std::set_intersection(incoming_labels.begin(), incoming_labels.end(), curr_node_labels.begin(),
                          curr_node_labels.end(), std::inserter(common_filters, common_filters.end()));
    if (common_filters.size() > 0)
    {
        // This is to reduce the repetitive calls. If common_filters size is > 0 ,
        // we dont need to check further for universal label
        return true;
    }
    if (_has_universal_label)
    {
        if (!search_invocation)
        {
            if (std::find(incoming_labels.begin(), incoming_labels.end(), _universal_label) != incoming_labels.end() ||
                std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.insert(_universal_label);
        }
        else
        {
            if (std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                common_filters.insert(_universal_label);
        }
    }
    return (common_filters.size() > 0);
}

template DISKANN_DLLEXPORT class InMemFilterStore<uint32_t>;
template DISKANN_DLLEXPORT class InMemFilterStore<uint16_t>;

} // namespace diskann