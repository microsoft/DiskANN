// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <any>
#include "tsl/robin_set.h"

namespace AnyWrapper
{
struct AnyContainerRef
{
    template <typename ContainerT> AnyContainerRef(ContainerT &container) : _data(&container)
    {
    }
    template <typename ContainerT> ContainerT &get()
    {
        auto set_ptr = std::any_cast<ContainerT *>(_data);
        return *set_ptr;
    }

  private:
    std::any _data;
};

struct AnyRobinSet
{
    template <typename T>
    AnyRobinSet(const tsl::robin_set<T> &robin_set) : _data(const_cast<tsl::robin_set<T> *>(&robin_set))
    {
    }

    template <typename T> const tsl::robin_set<T> &get() const
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(_data);
        return *set_ptr;
    }

    template <typename T> tsl::robin_set<T> &get()
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(_data);
        return *set_ptr;
    }

  private:
    std::any _data;
};

struct AnyVector
{
    template <typename T> AnyVector(const std::vector<T> &vector) : _data(const_cast<std::vector<T> *>(&vector))
    {
    }

    template <typename T> const std::vector<T> &get() const
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(_data);
        return *sharedVector;
    }

    template <typename T> std::vector<T> &get()
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(_data);
        return *sharedVector;
    }

  private:
    std::any _data;
};
} // namespace AnyWrapper
