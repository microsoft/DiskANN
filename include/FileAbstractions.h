#pragma once
#ifdef __NSG_WINDOWS__
#include <Windows.h>
typedef HANDLE FileHandle;
#else
typedef int FileHandle;
#endif



