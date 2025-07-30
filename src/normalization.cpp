#include "normalization.h"
#include <fstream>
#include <iostream>
#include "windows_customizations.h"

namespace diskann
{

// Global normalization configuration instance
DISKANN_DLLEXPORT NormalizationConfig g_normalization_config;

bool NormalizationConfig::load_from_file(const std::string& factors_file)
{
    std::ifstream file(factors_file);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open normalization factors file: " << factors_file << std::endl;
        return false;
    }
    
    // Read scale_factor and shift_factor from simple text file
    if (!(file >> scale_factor >> shift_factor))
    {
        std::cerr << "Warning: Could not read scale and shift factors from " << factors_file << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    enabled = true;
    
    std::cout << "Loaded normalization factors: scale=" << scale_factor 
              << ", shift=" << shift_factor << std::endl;
    
    return true;
}

bool initialize_normalization(const std::string& factors_file)
{
    return g_normalization_config.load_from_file(factors_file);
}

} // namespace diskann
