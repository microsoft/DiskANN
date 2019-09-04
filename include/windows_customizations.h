#pragma once

#ifdef __NSG_WINDOWS__
#define NSGDLLEXPORT __declspec(dllexport)
#else
#define NSGDLLEXPORT
#endif

