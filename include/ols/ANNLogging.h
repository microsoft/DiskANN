#pragma once

#include "IANNIndex.h"

namespace ANNIndex
{
    enum LogLevel
    {
        LL_Debug = 0,
        LL_Info,
        LL_Status,
        LL_Warning,
        LL_Error,
        LL_Assert,
        LL_Count
    };

#define KDTreeRNGLogging(level, ...) ANNIndex::ANNLogging(ANNIndex::AlgoNames[ANNIndex::AT_KDTreeRNG], (level), __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define DiskANNLogging(level, ...) ANNIndex::ANNLogging(ANNIndex::AlgoNames[ANNIndex::AT_DiskANN], (level), __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define IVFPQLogging(level, ...) ANNIndex::ANNLogging(ANNIndex::AlgoNames[ANNIndex::AT_IVFPQHNSW], (level), __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)


    void ANNLogging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...);
}
