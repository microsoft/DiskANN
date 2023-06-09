// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef DISKANN_PROGRAM_OPTIONS_UTILS_CPP
#define DISKANN_PROGRAM_OPTIONS_UTILS_CPP

#include <string.h>

namespace program_options_utils
{
/**
 * String appended to command-line tool help output to make the user aware that an argument is required
 */
inline const char required[] = "*";

const std::string make_required_param(const char *input)
{
    return std::string(required).append(" ").append(input);
}

const std::string make_program_description(const char *executable_name, const char *description)
{
    return std::string("\n")
        .append(description)
        .append("\n\n")
        .append("Arguments with ")
        .append(required)
        .append(" are required")
        .append("\n\n")
        .append("Usage: ")
        .append(executable_name)
        .append(" [OPTIONS]");
}
} // namespace program_options_utils

#endif // DISKANN_PROGRAM_OPTIONS_UTILS_CPP
