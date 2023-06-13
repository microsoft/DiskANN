// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef DISKANN_PROGRAM_OPTIONS_UTILS_CPP
#define DISKANN_PROGRAM_OPTIONS_UTILS_CPP

#include <string.h>

namespace program_options_utils
{
const std::string make_program_description(const char *executable_name, const char *description)
{
    return std::string("\n")
        .append(description)
        .append("\n\n")
        .append("Usage: ")
        .append(executable_name)
        .append(" [OPTIONS]");
}
} // namespace program_options_utils

#endif // DISKANN_PROGRAM_OPTIONS_UTILS_CPP
