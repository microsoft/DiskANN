#include "slot_manager.h"

namespace diskann
{
template <typename tag_t> SlotManager<tag_t>::SlotManager(const uint32_t capacity) : _capacity(capacity)
{
    _tag_to_location.reserve(capacity);
    _location_to_tag.reserve(capacity);
}

template <typename tag_t> location_t SlotManager<tag_t>::capacity() const
{
    return _capacity;
}
template <typename tag_t> location_t SlotManager<tag_t>::number_of_used_locations() const
{
    return _tags_to_location.size();
}

template <typename tag_t> location_t SlotManager<tag_t>::load(const std::string &filename)
{
    //We must load the delete set first because during save we don't remove deleted points.
    //Any point that is present in tags file but is also in the delete set will be discarded.
    load_delete_set(filename);
    load_tags(filename);
    _capacity = _tags_to_location.size();
    return _tags_to_location.size(); 
}

template <typename tag_t>
size_t SlotManager<tag_t>::save(const std::string& filename)
{
    size_t bytes_written = save_tags(filename);
    bytes_written += save_delete_set(filename);
    return bytes_written;
}

template <typename tag_t> void SlotManager<tag_t>::load_tags(const std::string &tag_filename)
{
    // REFACTOR: This had _enable_tags in an AND condition earlier.
    if (!file_exists(tag_filename))
    {
        std::stringstream ss;
        ss << "Tag file " << tag_filename << " does not exist !" << std::endl;
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // REFACTOR
    // if (!_enable_tags)
    //{
    //     diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
    //     return 0;
    // }

    size_t file_dim, file_num_points;
    TagT *tag_data;
#ifdef EXEC_ENV_OLS
    load_bin<TagT>(reader, tag_data, file_num_points, file_dim);
#else
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim);
#endif

    if (file_dim != 1)
    {
        std::stringstream stream;
        stream << "ERROR: Found " << file_dim << " dimensions for tags,"
               << "but tag file must have 1 dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        delete[] tag_data;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // REFACTOR: We used to put zeros for num_frozen_points at the end of the tag file
    //  during save and ignore them during load. Now we'll avoid it, so that way the
    //  slot manager need not know about num_frozen_points.
    const size_t num_data_points = file_num_points; //- _num_frozen_pts;
    _location_to_tag.reserve(num_data_points);
    _tag_to_location.reserve(num_data_points);
    for (uint32_t i = 0; i < (uint32_t)num_data_points; i++)
    {
        TagT tag = *(tag_data + i);
        if (_delete_set->find(i) == _delete_set->end())
        {
            _location_to_tag.set(i, tag);
            _tag_to_location[tag] = i;
        }
    }
    diskann::cout << "Tags loaded." << std::endl;
    delete[] tag_data;
    return file_num_points;
}

template <typename tag_t> void SlotManager<tag_t>::save_tags(const std::string &tags_filename)
{
    if (!_enable_tags)
    {
        diskann::cout << "Not saving tags as they are not enabled." << std::endl;
        return 0;
    }
    size_t tag_bytes_written;

    // REFACTOR
    //TagT *tag_data = new TagT[_nd + _num_frozen_pts];
    TagT *tag_data = new TagT[_tags_to_location.size()];
    for (uint32_t i = 0; i < _tags_to_location.size(); i++)
    {
        TagT tag;
        if (_location_to_tag.try_get(i, tag))
        {
            tag_data[i] = tag;
        }
        else
        {
            // point has been deleted
            // catering to future when tagT can be any type.
            std::memset((char *)&tag_data[i], 0, sizeof(TagT));
        }
    }

    // REFACTOR: We used to put zeros for num_frozen_points at the end of the tag file
    // and remove them during load. Avoiding it now.
    // if (_num_frozen_pts > 0)
    //{
    //     std::memset((char *)&tag_data[_start], 0, sizeof(TagT) * _num_frozen_pts);
    // }
    try
    {
        //REFACTOR
        //tag_bytes_written = save_bin<TagT>(tags_filename, tag_data, _nd + _num_frozen_pts, 1);
        tag_bytes_written = save_bin<TagT>(tags_filename, tag_data, _nd, 1);
    }
    catch (std::system_error &e)
    {
        throw FileException(tags_filename, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    delete[] tag_data;
    return tag_bytes_written;
}

template <typename tag_t> void SlotManager<tag_t>::load_delete_set(const std::string& filename)
{
    std::unique_ptr<uint32_t[]> delete_list;
    size_t npts, ndim;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint32_t>(reader, delete_list, npts, ndim);
#else
    diskann::load_bin<uint32_t>(filename, delete_list, npts, ndim);
#endif
    assert(ndim == 1);
    for (uint32_t i = 0; i < npts; i++)
    {
        _delete_set->insert(delete_list[i]);
    }
    return npts;
}
template <typename tag_t> void SlotManager<tag_t>::save_delete_set(const std::string &filename)
{
    if (_delete_set->size() == 0)
    {
        return 0;
    }
    std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set->size());
    uint32_t i = 0;
    for (auto &del : *_delete_set)
    {
        delete_list[i++] = del;
    }
    return save_bin<uint32_t>(filename, delete_list.get(), _delete_set->size(), 1);
}

template <typename tag_t> location_t SlotManager<tag_t>::resize(const location_t new_num_points)
{
    if (new_num_points > _tag_to_location.size())
    {
        _location_to_tag.reserve(new_num_points);
        _tag_to_location.reserve(new_num_points);
    }
    //REFACTOR TODO: It is not clear if we should support shrink as well, but currently, we will not.
    return _tag_to_location.size();
}
   
template <typename tag_t> location_t SlotManager<tag_t>::get_location_for_tag(const tag_t &tag)
{
  return _tag_to_location[tag];
}

template <typename tag_t> tag_t SlotManager<tag_t>::get_tag_at_location(location_t slot)
{
  return _location_to_tag[slot];
}
// Add a new tag into the slot manager. If the tag was added successfully,
// it fills the location of the tag in the "location" argument and returns
// Success. If the tag already exists, it returns TagAlreadyExists and if
// there is no space to add the tag, it returns MaxCapacityExceeded. In
// both these cases, 'location' contains an invalid value.
template <typename tag_t>
SlotManager<tag_t>::ErrorCode SlotManager<tag_t>::add_tag(const tag_t &tag, location_t &location);

// Delete a tag from the slot manager. If the tag was deleted successfully,
// it returns Success and 'location' contains the slot that was freed.
template <typename tag_t>
SlotManager<tag_t>::ErrorCode SlotManager<tag_t>::delete_tag(const tag_t &tag, location_t &location);

template <typename tag_t> bool SlotManager<tag_t>::exists(const tag_t &tag);

template <typename tag_t> void SlotManager<tag_t>::compact(std::vector<location_t> &new_locations);

// TODO: these are intrusive methods, but keeping them to make the port easier.
// Must revisit later.
template <typename tag_t> void SlotManager<tag_t>::get_delete_set(std::vector<location_t> &copy_of_delete_set);
template <typename tag_t> void SlotManager<tag_t>::clear_delete_set();

} // namespace diskann
