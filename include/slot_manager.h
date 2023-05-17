#pragma once

#include <memory>
#include "tsl/sparse_map.h"
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
    virtual location_t number_of_used_locations() const override;

    virtual void reposition_frozen_point(const location_t new_loc) override;
    virtual location_t get_frozen_point_location() const override;


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
    virtual ErrorCode add_tag(const tag_t &tag, location_t &location) override;

    // Delete a tag from the slot manager. If the tag was deleted successfully,
    // it returns Success and 'location' contains the slot that was freed.
    virtual ErrorCode delete_tag(const tag_t &tag, location_t &location) override;

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
    location_t _num_active_points;

    
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
