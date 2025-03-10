#include "log_utils.h"
#include <cstdlib>  // For getenv

std::ofstream& get_log_file() {
    static std::ofstream log_file;
    static bool initialized = false;

    if (!initialized) {
        const char* log_path = std::getenv("DISKANN_LOG_FILE");
        if (log_path) {
            log_file.open(log_path, std::ios::app);
        } else {
            log_file.open("diskann_default.log", std::ios::app);  // Fallback
        }
        initialized = true;
    }
    return log_file;
}
