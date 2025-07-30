#pragma once

#include <string>
#include <windows_customizations.h>

namespace diskann
{

/**
 * @brief Simple normalization configuration
 */
struct NormalizationConfig
{
    float scale_factor = 1.0f;
    float shift_factor = 0.0f;
    bool enabled = false;
    
    /**
     * @brief Load normalization factors from simple text file
     * Format: "scale_factor shift_factor" on one line
     * @param factors_file Path to the text file containing scale and shift factors
     * @return true if successfully loaded, false otherwise
     */
    bool load_from_file(const std::string& factors_file);
    
    /**
     * @brief Apply normalization to a distance value
     * @param distance Original distance value
     * @return Normalized distance value
     */
    inline float normalize_distance(float distance) const
    {
        if (!enabled) return distance;
        return (distance + shift_factor) * scale_factor;
    }
};

/**
 * @brief Global normalization configuration instance
 */
DISKANN_DLLEXPORT extern NormalizationConfig g_normalization_config;

/**
 * @brief Initialize normalization from factors file
 * @param factors_file Path to text file with scale and shift factors
 * @return true if successful, false otherwise
 */
DISKANN_DLLEXPORT bool initialize_normalization(const std::string& factors_file);

} // namespace diskann
