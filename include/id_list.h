// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <string>
#include "roaring.h"
#include "types.h"
#include "windows_customizations.h"
#include "distance.h"
#include <iterator>

namespace diskann
{

class AbstractIdList
{
  public:
    AbstractIdList() {   
    }

    virtual ~AbstractIdList() = default;

    virtual uint64_t size() = 0;

    virtual void add(const uint32_t val) = 0;

    virtual void intersect_list(const AbstractIdList &other) = 0;

    virtual void union_list(const AbstractIdList &other) = 0;

    virtual void* get_bitmap() const = 0;

  protected:

};

class RoaringIdList : public AbstractIdList
{
  public:
    RoaringIdList() {
      list = roaring_bitmap_create();
    }

    ~RoaringIdList() {
          roaring_bitmap_free(list);
    }

    uint64_t size() {
      return roaring_bitmap_get_cardinality(list);
    }

    void add(const uint32_t val) {
     roaring_bitmap_add(list, val); 
    }

    void intersect_list(const AbstractIdList &other) {
      roaring_bitmap_and_inplace(list, (roaring_bitmap_t*) (other.get_bitmap()));
    }

    void union_list(const AbstractIdList &other) {
      roaring_bitmap_or_inplace(list, (roaring_bitmap_t*) (other.get_bitmap()));
    }

    void* get_bitmap() const {
      return (void*) list;
    }

  protected:

    roaring_bitmap_t *list;
    
};

} // namespace diskann
