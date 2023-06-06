#pragma once

#include "types.h"

namespace diskann
{
  enum class SlotManagerErrorCode
  {
      Success = 0,
      TagAlreadyExists = -1,
      MaxCapacityExceeded = -2,
      TagDoesNotExist = -3
  };


  template <typename TagT> 
  class AbstractSlotManager
  {
    public:
      virtual ~AbstractSlotManager() = default;

      virtual location_t capacity() const = 0;
      virtual location_t num_frozen_points() const = 0;

      //This is the number of slots occupied including soft deleted points before compaction. After compaction
      // it is the number of active points
      virtual location_t num_used_slots() const = 0;

      //The number of points that have been soft deleted. 
      virtual location_t num_deleted_points() const = 0;

      //This is the number of slots available to insert points. In implementation terms, it will 
      //be the capacity - number_of_used_slots() + number_of_deleted_points(), before compaction.
      //After compaction, it'll be capacity - number_of_used_slots() OR 
      //capacity - number_of_active_points() - both will be the same. 
      virtual location_t num_available_slots() const = 0;

      //This is the number of slots occupied by active points, i.e., points that are not soft deleted.
      virtual location_t num_active_points() const = 0;

      virtual void reposition_frozen_points(const location_t new_loc) = 0;
      virtual location_t get_frozen_points_start_location() const = 0;
      
      //Returns the number of tags loaded from the file.
      virtual location_t load(const std::string& filename) = 0;

      //Returns the NUMBER OF BYTES written to the file.
      virtual size_t save(const std::string& filename) = 0;

      virtual location_t resize(const location_t new_num_points) = 0;

      virtual location_t get_location_for_tag(const TagT& tag) = 0;
      virtual TagT get_tag_at_location(const location_t slot) = 0;

      // Add a new tag into the slot manager. If the tag was added successfully,
      // it fills the location of the tag in the "location" argument and returns
      // Success. If the tag already exists, it returns TagAlreadyExists and if 
      // there is no space to add the tag, it returns MaxCapacityExceeded. In
      // both these cases, 'location' contains an invalid value.
      virtual SlotManagerErrorCode add_tag(const TagT& tag, location_t& location) = 0;

      //Delete a tag from the slot manager. If the tag was deleted successfully,
      //it returns Success and 'location' contains the slot that was freed. 
      virtual SlotManagerErrorCode delete_tag(const TagT &tag, location_t& location) = 0;

      virtual bool exists(const TagT& tag) = 0;

      virtual void compact(std::vector<location_t>& new_locations) = 0;

      //TODO: these are intrusive methods, but keeping them to make the port easier.
      //Must revisit later.
      virtual void get_delete_set(std::vector<location_t> &copy_of_delete_set) = 0;
      virtual void clear_delete_set() = 0;

      protected:
        AbstractSlotManager() = default;
        virtual void load_tags(const std::string &filename) = 0;
        virtual void load_delete_set(const std::string &filename) = 0;

        // Returns the number of bytes written to the file.
        virtual size_t save_tags(const std::string &filename) = 0;
        // Returns the number of bytes written to the file.
        virtual size_t save_delete_set(const std::string &filename) = 0;


  };
} // namespace diskann