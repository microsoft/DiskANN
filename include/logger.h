// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <functional>
#include <iostream>
#include "windows_customizations.h"

namespace diskann {
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cout;
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cerr;

  enum class DISKANN_DLLEXPORT LogLevel { LL_Info = 0, LL_Error, LL_Count };

#ifdef EXEC_ENV_OLS
  DISKANN_DLLEXPORT void SetCustomLogger(
      std::function<void(LogLevel, const char*)> logger);
#endif
}  // namespace diskann
