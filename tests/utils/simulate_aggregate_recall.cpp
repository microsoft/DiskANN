// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>

inline float aggregate_recall(const unsigned k_aggr, const unsigned k, const unsigned npart, unsigned *count,
                              const std::vector<float> &recalls)
{
    float found = 0;
    for (unsigned i = 0; i < npart; ++i)
    {
        size_t max_found = std::min(count[i], k);
        found += recalls[max_found - 1] * max_found;
    }
    return found / (float)k_aggr;
}

void simulate(const unsigned k_aggr, const unsigned k, const unsigned npart, const unsigned nsim,
              const std::vector<float> &recalls)
{
    std::random_device r;
    std::default_random_engine randeng(r());
    std::uniform_int_distribution<int> uniform_dist(0, npart - 1);

    unsigned *count = new unsigned[npart];
    double aggr_recall = 0;

    for (unsigned i = 0; i < nsim; ++i)
    {
        for (unsigned p = 0; p < npart; ++p)
        {
            count[p] = 0;
        }
        for (unsigned t = 0; t < k_aggr; ++t)
        {
            count[uniform_dist(randeng)]++;
        }
        aggr_recall += aggregate_recall(k_aggr, k, npart, count, recalls);
    }

    std::cout << "Aggregate recall is " << aggr_recall / (double)nsim << std::endl;
    delete[] count;
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cout << argv[0] << " k_aggregate k_out npart nsim recall@1 recall@2 ... recall@k" << std::endl;
        exit(-1);
    }

    const unsigned k_aggr = atoi(argv[1]);
    const unsigned k = atoi(argv[2]);
    const unsigned npart = atoi(argv[3]);
    const unsigned nsim = atoi(argv[4]);

    std::vector<float> recalls;
    for (int ctr = 5; ctr < argc; ctr++)
    {
        recalls.push_back(atof(argv[ctr]));
    }

    if (recalls.size() != k)
    {
        std::cerr << "Please input k numbers for recall@1, recall@2 .. recall@k" << std::endl;
    }
    if (k_aggr > npart * k)
    {
        std::cerr << "k_aggr must be <= k * npart" << std::endl;
        exit(-1);
    }
    if (nsim <= npart * k_aggr)
    {
        std::cerr << "Choose nsim > npart*k_aggr" << std::endl;
        exit(-1);
    }

    simulate(k_aggr, k, npart, nsim, recalls);

    return 0;
}
