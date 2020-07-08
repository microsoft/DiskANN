#pragma once

#include "APWrapper.h"
#include "IANNIndex.h"
#include "ANNLogging.h"

namespace ANNIndex
{
    DEFINE_CUSTOM_LOGID(ANNAlgoLibrary);

    void ANNLogging(const char* title, LogLevel level, const char* format, ...)
    {
        va_list args;
        va_start(args, format);
                                                                        
        APWrapper::LogMessageV2(__FILE__, __LINE__, __FUNCTION__,
            level == LL_Info? APWrapper::LogLevel_Info :
            level == LL_Warning ? APWrapper::LogLevel_Warning :
            level == LL_Error ? APWrapper::LogLevel_Error :
            APWrapper::LogLevel_Debug, LogId_ANNAlgoLibrary, title, format, args);

        va_end(args);
    }
}

