#pragma once

#include "windows_customizations.h"

namespace diskann
{
	namespace logging
	{
        enum LogLevel
        {
            Debug,
            Info,
            Status,
            Warning,
            Error,
            Assert
        };

        // Interface for a logger that can be used to log messages at various levels.
        class ILogger
        {
        public:
            virtual ~ILogger() = default;

            // Log a message associated with a specific level, title, file name, function, and line number.
            virtual void Write(char const *filename, 
                               char const *function, 
                               unsigned lineNumber,
                               LogLevel level, 
                               char const *title, 
                               char const *message) = 0;

            // Abort the program.
            virtual void Abort() = 0;
        };

        DISKANN_DLLEXPORT void RegisterLogger(ILogger *logger);
    }
}