// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef _WINDOWS
#define DISKANN_DLLEXPORT __declspec(dllexport)
#define DISKANN_DLLIMPORT __declspec(dllimport)
#else
#define DISKANN_DLLEXPORT
#define DISKANN_DLLIMPORT
#endif
