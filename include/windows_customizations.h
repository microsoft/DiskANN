// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef _WINDOWS

#if defined(DISKANN_STATIC_LIB)
// Static-library build/consumer (e.g. the unit tests): the internal symbols
// are compiled straight into the linking target, so no dllimport/dllexport
// decoration is needed. Checked first so it wins over _WINDLL.
#define DISKANN_DLLEXPORT
#elif defined(_WINDLL)
#define DISKANN_DLLEXPORT __declspec(dllexport)
#else
#define DISKANN_DLLEXPORT __declspec(dllimport)
#endif

#else
#define DISKANN_DLLEXPORT
#endif
