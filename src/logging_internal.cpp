#include "logging.h"

#include <cstdarg>
#include <sstream>

// This is not getting exposed through the DLL.
#include "logging_internal.h"
#include "logger.h"

namespace diskann
{
	namespace logging
	{
        namespace
        {
            // Formatting functions for log messages.
		    constexpr unsigned c_messageBufferSizeOnStack = 4096;

            // A marker to be placed at the end of the truncated log lines.
            const char c_truncatedMessage[] = "[TEXT TRUNCATED]";

            template <size_t SIZE>
            void FormatMessage(char (&buffer)[SIZE], char const *format, va_list args)
            {
                static_assert(SIZE > _countof(c_truncatedMessage), "The message buffer is too short for truncation marker");

                const int result = vsnprintf_s(buffer, _TRUNCATE, format, args);

                // If truncation occurred, place the truncation marker at the end of the
                // buffer. Note that _countof also counts the terminating '\0'.
                if (result == -1)
                {
                    const unsigned position = SIZE - _countof(c_truncatedMessage);
                    memcpy(&buffer[position], &c_truncatedMessage[0], _countof(c_truncatedMessage));
                }
            }
        }

        //
        // ConsoleLogger implementation.
        //
        void ConsoleLogger::Write(char const *filename, char const *function, unsigned lineNumber, LogLevel level,
                               char const *title, char const *message)
        {
            auto& output = level == LogLevel::Error || level == LogLevel::Assert ? diskann::cerr : diskann::cout;
            output << "[" << filename << ": " << function << ": " << lineNumber << "] " << title << ": " << message
                   << std::endl;
        }

        void ConsoleLogger::Abort()
        {
            // No-op.
        }

        // By default console logger is used.
        static ConsoleLogger consoleLogger;
        static ILogger *g_logger = &consoleLogger;

        // Function to register a custom logger.
        void RegisterLogger(ILogger *logger)
        {
            g_logger = (logger != nullptr) ? logger : &consoleLogger;
        }

        // Logging implementation.
        void LogImpl(char const *filename, 
                     char const *function, 
                     unsigned lineNumber, 
                     LogLevel level, 
                     char const *title,
                     char const *format, ...)
        {
            char message[c_messageBufferSizeOnStack];

            va_list arguments;
            va_start(arguments, format);
            FormatMessage(message, format, arguments);
            va_end(arguments);

            g_logger->Write(filename, function, lineNumber, level, title, message);
        }
    }
}