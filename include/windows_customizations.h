#pragma once

#ifdef _WINDOWS
#define NSGDLLEXPORT __declspec(dllexport)
#else
#define NSGDLLEXPORT
#endif
