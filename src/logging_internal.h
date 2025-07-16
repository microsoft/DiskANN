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

        // A no-op logger that does not log anything.
        class NullLogger : public ILogger
        {
        public:
            virtual void Write(char const *filename, char const *function, unsigned lineNumber, LogLevel level,
                               char const *title, char const *message) override;

            virtual void Abort() override;
        };

        // Generates a log message for a specified title, source file, function,
        // and line number.
        // The format arument and those that follow are passed to sprintf() to
        // create the log message.
        // This method is thread safe.
        void LogImpl(char const *filename, char const *function, unsigned lineNumber, LogLevel level, char const *title,
                     char const *format, ...);

        // Logs a message and then throws an exception. The condition parameter
        // has the text of the condition causing the exception to be thrown.
        // Its value is typically set with the text of the first argument to the
        // LogThrowAssert() macro. Any arguments after condition are passed to
        // sprintf() to create the rest of the log message.
        // This method is thread safe.
        _declspec(noreturn) void LogThrowImpl(char const *filename, char const *function, unsigned lineNumber,
                                              char const *condition, char const *format, ...);

        // Logs a message and then instructs the logger to terminate the
        // program. The condition parameter has the text of the condition
        // causing the program to shut down. Its value is typically set with
        // the text of the first argument to the LogAssert() macro. Any
        // arguments after condition are passed to sprintf() to create the rest
        // of the log message.
        // This method is thread safe.
        void LogAbortImpl(char const *filename, char const *function, unsigned lineNumber, char const *condition);
        void LogAbortImpl(char const *filename, char const *function, unsigned lineNumber, char const *condition,
                          char const *format, ...);

        // Useful in scenerio where there is a throw assert and the message needs
        // to be formatted but logging of that error message is not required.
        void ThrowImpl(char const *format, ...);
    }
}