// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "any_wrappers.h"
#include <any>
#include <cstddef>
#include <cstdint>

namespace diskann {
typedef uint32_t location_t;

using DataType = std::any;
using TagType = std::any;
using LabelType = std::any;
using TagVector = AnyWrapper::AnyVector;
using DataVector = AnyWrapper::AnyVector;
using Labelvector = AnyWrapper::AnyVector;
using TagRobinSet = AnyWrapper::AnyRobinSet;
} // namespace diskann
