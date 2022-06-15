// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "windows_customizations.h"

namespace diskann {
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cout;
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cerr;
}  // namespace diskann
