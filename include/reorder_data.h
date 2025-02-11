#pragma once
#include <cstdint>

namespace diskann
{

struct reorder_data
{
    std::uint8_t *data = nullptr;
    size_t size = 0;
};

}