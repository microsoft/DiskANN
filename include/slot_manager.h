#pragma once

#include <memory>
#include "tsl/sparse_map.h"
#include "tsl/robin_set.h"
#include "natural_number_map.h"
#include "natural_number_set.h"

#include "abstract_slot_manager.h"

namespace diskann
{
template <typename tag_t> class SlotManager : AbstractSlotManager<tag_t>
{
  public:
    SlotManager(const location_t capacity, const location_t num_frozen_points);

    virtual location_t capacity() const override;
    virtual location_t num_frozen_points() const override;

    // This is the number of slots occupied including soft deleted points before compaction. After compaction
    //  it is the number of active points
    virtual location_t num_used_slots() const override;

    // The number of points that have been soft deleted.
    virtual location_t num_deleted_points() const override;

    // This is the number of slots available to insert points. In implementation terms, it will
    // be the capacity - num_used_slots() + num_deleted_points(), before compaction.
    // After compaction, it'll be capacity - num_used_slots() OR
    // capacity - num_active_points() - both will be the same.
    virtual location_t num_available_slots() const override;

    // This is the number of slots occupied by active points, i.e., points that are not soft deleted.
    virtual location_t num_active_points() const override;


    virtual void reposition_frozen_points(const location_t new_loc) override;
    virtual location_t get_frozen_points_start_location() const override;


    virtual location_t load(const std::string &filename) override;
    virtual size_t save(const std::string &filename) override;

    virtual location_t resize(const location_t new_num_points) override;

    virtual location_t get_location_for_tag(const tag_t &tag) override;
    virtual tag_t get_tag_at_location(const location_t slot) override;

    // Add a new tag into the slot manager. If the tag was added successfully,
    // it fills the location of the tag in the "location" argument and returns
    // Success. If the tag already exists, it returns TagAlreadyExists and if
    // there is no space to add the tag, it returns MaxCapacityExceeded. In
    // both these cases, 'location' contains an invalid value.
    virtual SlotManagerErrorCode add_tag(const tag_t &tag, location_t &location) override;

    // Delete a tag from the slot manager. If the tag was deleted successfully,
    // it returns Success and 'location' contains the slot that was freed.
    virtual SlotManagerErrorCode delete_tag(const tag_t &tag, location_t &location) override;

    virtual bool exists(const tag_t &tag) override;

    virtual void compact(std::vector<location_t> &new_locations) override;

    // TODO: these are intrusive methods, but keeping them to make the port easier.
    // Must revisit later.
    virtual void get_delete_set(std::vector<location_t> &copy_of_delete_set) override;
    virtual void clear_delete_set() override;

protected:
    virtual void load_tags(const std::string &filename) override;
    virtual void load_delete_set(const std::string &filename) override;

    virtual void save_tags(const std::string &filename) override;
    virtual void save_delete_set(const std::string &filename) override;


  private:
    location_t _capacity;
    location_t _num_frozen_points;
    location_t _next_available_slot; //equivalent of _nd in index.cpp

    
    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    tsl::sparse_map<tag_t, uint32_t> _tag_to_location;
    natural_number_map<uint32_t, tag_t> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<uint32_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<uint32_t>> _delete_set;
};
} // namespace diskann
