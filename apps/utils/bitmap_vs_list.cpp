// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>
#include <random>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "roaring.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{

    uint32_t maxN;
    float p1,p2;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        po::options_description optional_configs("Optional");

        required_configs.add_options()("maxN",
                                       po::value<uint32_t>(&maxN)->default_value(10000000),
                                       "maxN");
        required_configs.add_options()("p1",
                                       po::value<float>(&p1)->default_value(0.1),
                                       "p1");
        required_configs.add_options()("p2",
                                       po::value<float>(&p2)->default_value(0.1),
                                       "p2");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }


    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1.0);

//    uint32_t maxN = 10000000;
//    float p1_val = 0.1;
//    float p2_val = 0.1;
    std::vector<uint32_t> v1;
    std::vector<uint32_t> v2;
    for (uint32_t i = 0; i < maxN; i++) {        
        float tau = dis(gen);
        if (tau < p1)
            v1.push_back(i);
    }
    for (uint32_t i = 0; i < maxN; i++) {        
        float tau = dis(gen);
        if (tau < p2)
            v2.push_back(i);
    }

    std::cout<<"sizes of v1, v2: " << v1.size() << " " << v2.size() << std::endl;

    roaring_bitmap_t *r1 = roaring_bitmap_create();
    roaring_bitmap_t *r2 = roaring_bitmap_create();
    for (auto &x : v1)
        roaring_bitmap_add(r1, x);
    for (auto &x : v2)
        roaring_bitmap_add(r2, x);

    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
//        auto intersect = roaring_bitmap_create();
        auto intersect = roaring_bitmap_and(r1, r2);
        if (i == 99) std::cout<<roaring_bitmap_get_cardinality(intersect) << " is the size of intersection computed using roaring." << std::endl;
        roaring_bitmap_free(intersect);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "set intersection computation using roaring bitmap time:" << diff.count()/100 << std::endl;


    s = std::chrono::high_resolution_clock::now();
    uint64_t count = 0;
    for (int i = 0; i < 100; i++) {
    for (auto &y : v2) {
        bool flag = roaring_bitmap_contains(r1, y);
        if (flag)
            count++;
    }
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "membership lookup time using roaring:" << diff.count()/(v2.size()*100) << std::endl;


    s = std::chrono::high_resolution_clock::now();
    diff = std::chrono::high_resolution_clock::now() - s;
    for (int i = 0; i < 100; i++) {
        auto intersect = roaring_bitmap_create();
        for (auto &y : v2) {
            bool flag = roaring_bitmap_contains(r1, y);
            if (flag)
                roaring_bitmap_add(intersect, y);
        }
        roaring_bitmap_free(intersect);        
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "set intersection computation using roaring lookups time:" << diff.count()/(100) << std::endl;


    tsl::robin_set<uint32_t> a;
    for (auto &x: v1)
        a.insert(x);
    
    s = std::chrono::high_resolution_clock::now();
    count=0;    
    for (int i = 0; i < 100; i++) {
        for (auto &y : v2) {
            bool flag = a.find(y) != a.end() ? true : false;
            if (flag)
                count++;
            }
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout<<"intersection count per robin_set: " << count/100 << std::endl;
    std::cout << "tsl robin membership check time:" << diff.count()/(v2.size()*100) << std::endl;


    s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        tsl::robin_set<uint32_t> robin_intersect;    
        for (auto &y : v2) {
            bool flag = a.find(y) != a.end() ? true : false;
            if (flag)
                robin_intersect.insert(y);
            }
            if (i == 99)
                std::cout<<robin_intersect.size() << std::endl;
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout<< "intersection computation time via iterative tsl lookups:" << diff.count()/100 << std::endl;


    s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        std::vector<uint32_t> common_filters;
        std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(common_filters));
        if (i==99)
        std::cout<< common_filters.size() << std::endl;
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "intersection computation using std set intersection time:" << diff.count()/100 << std::endl;

    roaring_bitmap_free(r1);
    roaring_bitmap_free(r2);
    return 0;

}
