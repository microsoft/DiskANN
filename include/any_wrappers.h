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
struct AnyRobinSet
{
    template <typename T>
    AnyRobinSet(const tsl::robin_set<T> &robin_set) : data(const_cast<tsl::robin_set<T> *>(&robin_set))
    {
    }

    template <typename T> const tsl::robin_set<T> &get() const
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(&data);
        if (set_ptr)
        {
            return *(*set_ptr);
        }

        throw std::bad_any_cast();
    }

    template <typename T> tsl::robin_set<T> &get()
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(&data);
        if (set_ptr)
        {
            return *(*set_ptr);
        }

        throw std::bad_any_cast();
    }

  private:
    std::any data;
};
struct AnyVector
{
    template <typename T> AnyVector(const std::vector<T> &vector) : data(const_cast<std::vector<T> *>(&vector))
    {
    }

    template <typename T> const std::vector<T> &get() const
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(&data);
        if (sharedVector)
        {
            return *(*sharedVector);
        }

        throw std::bad_any_cast();
    }

    template <typename T> std::vector<T> &get()
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(&data);
        if (sharedVector)
        {
            return *(*sharedVector);
        }

        throw std::bad_any_cast();
    }

  private:
    std::any data;
};
} // namespace AnyWrapper
