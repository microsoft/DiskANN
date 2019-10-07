#pragma once

#ifdef _WINDOWS
#define DISKANN_DLLEXPORT __declspec(dllexport)
#else
#define DISKANN_DLLEXPORT
#endif
