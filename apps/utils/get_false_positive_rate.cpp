// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <boost/dynamic_bitset.hpp>
#include <unordered_map>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 10000000
#define ALIGNMENT 512

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set;
typedef std::string path;

namespace po = boost::program_options;

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}


inline void parse_base_label_file(const std::string &map_file,
                                      std::vector<tsl::robin_set<std::string>> &pts_to_labels, uint32_t start_id = 0)
{
    pts_to_labels.clear();
    std::ifstream infile(map_file);
    std::string line, token;
    std::set<std::string> labels;
    infile.clear();
    infile.seekg(0, std::ios::beg);
    uint32_t line_no = 0;
    while (std::getline(infile, line))
    { 
        if (line_no < start_id) {
            line_no++;
            continue;
        }
        line_no++;
        std::istringstream iss(line);
        tsl::robin_set<std::string> lbls;

        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            lbls.insert(token);
            labels.insert(token);
        }
//        std::sort(lbls.begin(), lbls.end());
        pts_to_labels.push_back(lbls);
        if (pts_to_labels.size() >= PARTSIZE)
        break;
    }
    std::cout << "Identified " << labels.size() << " distinct label(s), and populated labels for "
              << pts_to_labels.size() << " points" << std::endl;
}

// outer vector is # queries,  inner vector is size of the AND predicate
inline void parse_query_label_file(const std::string &query_label_file,
                                      std::vector<std::vector<std::string>> &query_labels)
{
    query_labels.clear();
    std::ifstream infile(query_label_file);
    std::string line, token;
    std::set<std::string> labels;
    infile.clear();
    infile.seekg(0, std::ios::beg);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<std::string> lbls(0);

        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, '&'))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            lbls.push_back(token);
            labels.insert(token);
        }
//        std::sort(lbls.begin(), lbls.end());
        query_labels.push_back(lbls);
    }
    std::cout << "Identified " << labels.size() << " distinct label(s), and populated labels for "
              << query_labels.size() << " queries" << std::endl;
}


//template<typename A, typename B>
// add UNIVERSAL LABEL SUPPORT
int identify_matching_points(const std::string &base, const size_t start_id, const std::string &query, const std::string &unv_label, std::vector<boost::dynamic_bitset<>> &matching_points, std::vector<std::pair<uint32_t, uint32_t>> &query_stats) {
    std::vector<tsl::robin_set<std::string>> base_labels;
    std::vector<std::vector<std::string>> query_labels;
    parse_base_label_file(base, base_labels, start_id);
    parse_query_label_file(query, query_labels);
    matching_points.clear();
    uint32_t num_query = query_labels.size();
    uint32_t num_base = base_labels.size();
    matching_points.resize(num_query);
    for (auto &x : matching_points)
        x.resize(num_base);
    std::cout<<"Starting to identify matching points "<< std::endl;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 128)
    for (uint32_t i = 0; i < num_query; i++) {
//        if (i % 100 == 0)
//        std::cout<<"."<< std::flush;
//        tsl::robin_set<uint32_t> matches;
        for (uint32_t j = 0; j < num_base; j++) {
            bool pass = true;
            if (unv_label.empty() || (base_labels[j].find(unv_label) == base_labels[j].end())) {
            for (uint32_t k = 0; k < query_labels[i].size(); k++) {
                if (base_labels[j].find(query_labels[i][k]) == base_labels[j].end()) {
                    pass = false;
                    break;
                }
            }
            }
            if (pass) {
                matching_points[i][j] = 1;
                query_stats[i].second++;
            }
        }
    }
 std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  std::cout << "It took me " << time_span.count() << " seconds." << std::endl;    
    return 0;
}



void get_fp_rate(const std::string &base_labels, const std::string &base_labels_bloom, const std::string &query_labels,const std::string &query_labels_bloom, const std::string &unv_label,
                                                                            size_t &nqueries, size_t &npoints)
{
//    int num_parts = get_num_parts<T>(base_labels.c_str());
    uint32_t num_parts =
        (npoints % PARTSIZE) == 0 ? npoints / PARTSIZE : (uint32_t)std::floor(npoints / PARTSIZE) + 1;

    std::vector<std::vector<std::pair<uint32_t, float>>> res(nqueries);
    std::vector<std::pair<uint32_t, uint32_t>> query_stats(nqueries);
    for (uint32_t i = 0; i < nqueries; i++) {
        query_stats[i].first = i;
        query_stats[i].second = 0;
    }

    std::vector<uint32_t> corrects(nqueries, 0);
    std::vector<uint32_t> positives(nqueries, 0);
    for (int p = 0; p < num_parts; p++)
    {
        size_t start_id = p * PARTSIZE;
//        load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);
        size_t end_id = start_id + npoints;

        std::vector<boost::dynamic_bitset<>> matching_points;
        std::vector<boost::dynamic_bitset<>> matching_points_bloom;
        identify_matching_points(base_labels, start_id, query_labels, unv_label, matching_points, query_stats);
        identify_matching_points(base_labels_bloom, start_id, query_labels_bloom, unv_label, matching_points_bloom, query_stats);
        for (size_t i = 0; i < nqueries; i++)
        {
            corrects[i] += matching_points[i].count();
            positives[i] += matching_points_bloom[i].count();
        }

    }
    float fp_rate = 0;
    uint32_t good_queries = 0;
        for (size_t i = 0; i < nqueries; i++)
        {
            if (corrects[i] > 0) {
                fp_rate += (1.0*positives[i])/(1.0*corrects[i]);
                good_queries++;
            }
            if (corrects[i] > positives[i]) {
                std::cout<<"ERROR! False negative(s) detected." << std::endl;
            }
        }

    fp_rate /= good_queries;
    std::cout<<"FP rate: " << fp_rate << std::endl;
};

uint64_t get_label_file_num_pts(std::string file) {
    uint64_t num_lines = 0;
    std::ifstream infile(file);
    std::string line, token;
    std::set<std::string> labels;
    infile.clear();
    infile.seekg(0, std::ios::beg);
    while (std::getline(infile, line))
    {
        num_lines++;
    }
    std::cout<<"Identified " << num_lines <<" lines in " << file << std::endl;
    return num_lines;
}

// add UNIVERSAL LABEL SUPPORT
int aux_main(const std::string &base_labels,const std::string &base_labels_bloom, const std::string &query_labels, const std::string &query_labels_bloom,const std::string &unv_label, const std::string &tags_file = std::string(""))
{
    size_t npoints, nqueries, dim;

    nqueries = get_label_file_num_pts(query_labels);
    npoints = get_label_file_num_pts(base_labels);

    if (nqueries > PARTSIZE)
        std::cerr << "WARNING: #Queries provided (" << nqueries << ") is greater than " << PARTSIZE
                  << ". Computing GT only for the first " << PARTSIZE << " queries." << std::endl;

    // load tags
    const bool tags_enabled = tags_file.empty() ? false : true;



    get_fp_rate(base_labels, base_labels_bloom, query_labels, query_labels_bloom, unv_label, nqueries, npoints);

    return 0;
}

int main(int argc, char **argv)
{
    std::string base_labels, query_labels, unv_label, base_labels_bloom, query_labels_bloom;

    try
    {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");

        desc.add_options()("base_labels", po::value<std::string>(&base_labels)->required(),
                           "File containing the base labels");
        desc.add_options()("query_labels", po::value<std::string>(&query_labels)->required(),
                           "File containing the query labels");
        desc.add_options()("base_labels_bloom", po::value<std::string>(&base_labels_bloom)->required(),
                           "File containing the base labels bloom");
        desc.add_options()("query_labels_bloom", po::value<std::string>(&query_labels_bloom)->required(),
                           "File containing the query labels bloom");
        desc.add_options()("universal_label", po::value<std::string>(&unv_label)->default_value(std::string()),
                           "universal_label value");

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

    try
    {
            aux_main(base_labels,base_labels_bloom, query_labels, query_labels_bloom, unv_label);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Failed." << std::endl;
        return -1;
    }
}
