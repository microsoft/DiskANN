#include "logging.h"

// This is not getting exposed through the DLL.
#include "logging_internal.h"

#include <sstream>
#include <cstdarg>

namespace diskann
{
namespace logging
{
    void NullLogger::Write(char const *filename, char const *function, unsigned lineNumber, LogLevel level,
                         char const *title, char const *message)
    {
        // No-op.
    }

    void NullLogger::Abort()
    {
        // No-op.
    }

    constexpr unsigned c_messageBufferSizeOnStack = 4096;

    static NullLogger g_nullLogger;

    static ILogger *g_logger = &g_nullLogger;

    void RegisterLogger(ILogger *logger)
    {
        g_logger = (logger != nullptr) ? logger : &g_nullLogger;
    }

    namespace
    {

        // A marker to be placed at the end of the truncated log lines.
        const char c_truncatedMessage[] = "[TEXT TRUNCATED]";

        template <size_t SIZE> void FormatMessage(char (&buffer)[SIZE], char const *format, va_list args)
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
    } // namespace

    void LogImpl(char const *filename, char const *function, unsigned lineNumber, LogLevel level, char const *title,
                 char const *format, ...)
    {
        char message[c_messageBufferSizeOnStack];

        va_list arguments;
        va_start(arguments, format);
        FormatMessage(message, format, arguments);
        va_end(arguments);

        g_logger->Write(filename, function, lineNumber, level, title, message);
    }

    void LogThrowImpl(char const *filename, char const *function, unsigned lineNumber, char const *condition,
                      char const *format, ...)
    {
        char message[c_messageBufferSizeOnStack];

        va_list arguments;
        va_start(arguments, format);
        FormatMessage(message, format, arguments);
        va_end(arguments);

        LogImpl(filename, function, lineNumber, logging::Error, "LogThrow", "[%s] %s", condition, &message[0]);

        throw std::runtime_error(&message[0]);
    }

    void ThrowImpl(char const *format, ...)
    {
        char message[c_messageBufferSizeOnStack];

        va_list arguments;
        va_start(arguments, format);
        FormatMessage(message, format, arguments);
        va_end(arguments);

        throw std::runtime_error(&message[0]);
    }

    void LogAbortImpl(char const *filename, char const *function, unsigned lineNumber, char const *condition)
    {
        std::stringstream sstream;
        sstream << condition << " failed";

        g_logger->Write(filename, function, lineNumber, logging::Assert, "LogAbort", sstream.str().c_str());
        g_logger->Abort();
    }

    void LogAbortImpl(char const *filename, char const *function, unsigned lineNumber, char const *condition,
                      char const *format, ...)
    {
        char message[c_messageBufferSizeOnStack];
        va_list arguments;
        va_start(arguments, format);

        vsprintf_s(message, sizeof(message), format, arguments);
        va_end(arguments);

        std::stringstream sstream;
        sstream << condition << " failed: " << message;

        g_logger->Write(filename, function, lineNumber, logging::Assert, "LogAbort", sstream.str().c_str());
        g_logger->Abort();
    }
}
}