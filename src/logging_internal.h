#include "logging.h"

namespace diskann
{
    namespace logging
    {
        // Macro that logs a message. The macro gathers the source file name and the
        // line number. The caller may specify optional arguments that will be passed to
        // sprintf() to create the rest of the log message.
        #define Log(level, title, format, ...)                                                                                \
            logging::LogImpl(__FILE__, __FUNCTION__, __LINE__, level, title, format, __VA_ARGS__)

        // Generates a log message for a specified title, source file, function,
        // and line number.
        // The format arument and those that follow are passed to sprintf() to
        // create the log message.
        // This method is thread safe.
        void LogImpl(char const* filename, 
                     char const* function, 
                     unsigned lineNumber, 
                     LogLevel level, 
                     char const* title,
                     char const* format, ...);

        // Implementation of a logger that outputs to diskann::cout and diskann::cerr.
        class ConsoleLogger : public ILogger
        {
        public:
            // ILogger interface implementation.
            virtual void Write(char const* filename, 
                               char const* function, 
                               unsigned lineNumber, 
                               LogLevel level,
                               char const* title, 
                               char const* message) override;

            virtual void Abort() override;
        };
    }
}