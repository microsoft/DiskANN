#pragma once

#include "types.h"

namespace diskann
{
  template <typename TagT> 
  class AbstractSlotManager
  {
    public:
      enum ErrorCode
      {
         Success = 0,
         TagAlreadyExists = -1,
         MaxCapacityExceeded = -2,
         TagDoesNotExist = -3

      };

      virtual ~AbstractSlotManager() = default;

      virtual location_t capacity() const = 0;
      virtual location_t number_of_frozen_points() const = 0;
      virtual location_t number_of_used_locations() const = 0;
      virtual location_t number_of_deleted_points() const = 0;
      virtual location_t number_of_available_slots() const = 0;

      virtual void reposition_frozen_point(const location_t new_loc) = 0;
      virtual location_t get_frozen_point_location() const = 0;
      
      //Returns the number of tags loaded from the file.
      virtual location_t load(const std::string& filename) = 0;

      //Returns the NUMBER OF BYTES written to the file.
      virtual size_t save(const std::string& filename) = 0;

      virtual location_t resize(const location_t new_num_points) = 0;

      virtual location_t get_location_for_tag(const TagT& tag) = 0;
      virtual TagT get_tag_at_location(location_t slot) = 0;

      // Add a new tag into the slot manager. If the tag was added successfully,
      // it fills the location of the tag in the "location" argument and returns
      // Success. If the tag already exists, it returns TagAlreadyExists and if 
      // there is no space to add the tag, it returns MaxCapacityExceeded. In
      // both these cases, 'location' contains an invalid value.
      virtual ErrorCode add_tag(const TagT& tag, location_t& location) = 0;

      //Delete a tag from the slot manager. If the tag was deleted successfully,
      //it returns Success and 'location' contains the slot that was freed. 
      virtual ErrorCode delete_tag(const TagT &tag, location_t& location) = 0;

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