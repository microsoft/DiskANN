// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cstddef>
#include <any>

namespace diskann
{
typedef uint32_t location_t;

using DataType = std::any;
using TagType = std::any;
using LabelType = std::any;

} // namespace diskann