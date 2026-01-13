#include <boost/program_options.hpp>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rabitq.h"
#include "utils.h"

namespace po = boost::program_options;

namespace
{
#pragma pack(push, 1)
struct RaBitQReorderHeader
{
    char magic[8];
    uint32_t version;
    uint32_t metric;
    uint32_t nb_bits;
    #error "build_rabitq_reorder_codes has been removed (RaBitQ reorder prefilter deprecated). Use build_disk_index with --build_rabitq_main_codes instead."
    uint64_t num_points;
