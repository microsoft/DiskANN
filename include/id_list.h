// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "roaring.hh"
#include "types.h"
#include "windows_customizations.h"
#include "distance.h"
#include <iterator>

namespace diskann
{

class AbstractIdList
{
  public:
    AbstractIdList()
    {
    }

    AbstractIdList(uint32_t size, uint32_t *vals)
    {
    }

    virtual ~AbstractIdList() = default;

    virtual uint64_t size() = 0;

    virtual void add(const uint32_t val) = 0;

    virtual void copy_from(const AbstractIdList &other) = 0;

    virtual void intersect_list(const AbstractIdList &other) = 0;

    virtual void union_list(const AbstractIdList &other) = 0;

    virtual roaring::Roaring get_list() = 0;

  protected:
};

class RoaringIdList : public AbstractIdList
{
  public:
    RoaringIdList()
    {
    }

    RoaringIdList(const RoaringIdList &d) : AbstractIdList(d)
    {
        list = d.list;
        //    std::cout<<"here" ;
    }

    RoaringIdList(uint32_t size, uint32_t *vals)
    {
        list = roaring::Roaring(size, vals);
    }

    ~RoaringIdList()
    {
    }

    uint64_t size()
    {
        return list.cardinality();
    }

    void add(const uint32_t val)
    {
        list.add(val);
    }

    void copy_from(const AbstractIdList &other)
    {
        const RoaringIdList &other_r = dynamic_cast<const RoaringIdList &>(other);
        list = other_r.list;
    }

    void intersect_list(const AbstractIdList &other)
    {
        const RoaringIdList &other_r = dynamic_cast<const RoaringIdList &>(other);
        list &= other_r.list;
    }

    void union_list(const AbstractIdList &other)
    {
        const RoaringIdList &other_r = dynamic_cast<const RoaringIdList &>(other);
        list |= other_r.list;
    }

    roaring::Roaring get_list()
    {
        return list;
    }

    // TODO: make this protected again and remove external usage of list
    //  protected:
    roaring::Roaring list;
};

} // namespace diskann
