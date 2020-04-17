#pragma once

#include "IANNIndex.h"

namespace ANNIndex
{
    enum class LogLevel
    {
        LL_Debug = 0,
        LL_Info,
        LL_Status,
        LL_Warning,
        LL_Error,
        LL_Assert,
        LL_Count
    };

#define KDTreeRNGLogging(level, ...) ANNIndex::ANNLogging(ANNIndex::AlgoNames[ANNIndex::AT_KDTreeRNG], (level),  __VA_ARGS__)
#define RandNSGLogging(level, ...) ANNIndex::ANNLogging(ANNIndex::AlgoNames[ANNIndex::AT_RandNSG], (level),  __VA_ARGS__)

    void ANNLogging(const char* title, LogLevel level, const char* format, ...);
}
