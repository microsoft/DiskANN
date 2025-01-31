#include <iostream>
#include <cstdlib>
#include "windows_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include "utils.h"
#include "timer.h"
#include <omp.h>
#include <random>

using namespace std;
using namespace diskann;

#define SECTOR_LEN 4096
#define TOTAL_READS 1000000

void do_reads(WindowsAlignedFileReader* reader, vector<AlignedRead>& read_reqs, uniform_int_distribution<> &distrib, mt19937 &gen)
{
    auto ctx = reader->get_ctx();

    // Modify read requests
    for (auto& read_req : read_reqs)
    {
        long long int sector_id = distrib(gen);
        read_req.offset = sector_id * SECTOR_LEN;
    }

    reader->read(read_reqs, ctx, false);
}

void do_reads_in_batches_of(int batches_of)
{
    string file_name = "F:\\indices\\turing_10m\\disk_index_disk.index";
    auto reader = new WindowsAlignedFileReader();
    reader->open(file_name.c_str());
    char* buf = nullptr;
    alloc_aligned((void**)&buf, batches_of * SECTOR_LEN, SECTOR_LEN);
    reader->register_thread();

    std::vector<AlignedRead> read_reqs;
    read_reqs.reserve(batches_of);

    // create read requests
    for (size_t i = 0; i < batches_of; ++i)
    {
        AlignedRead read;
        read.len = SECTOR_LEN;
        read.buf = buf + i * SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // Initialize a random number generator
    uniform_int_distribution<> distrib(0, 1650000);
    random_device rd;
    mt19937 gen(rd());

    int no_of_reads = TOTAL_READS / batches_of;
    Timer timer;
    for (int i = 0; i < no_of_reads; i++)
    {
        do_reads(reader, read_reqs, distrib, gen);
    }
    // cout << "Time taken to read in microseconds: " << timer.elapsed() << endl;
    cout << timer.elapsed() << endl;

    reader->close();
}

int main(int argc, char* argv[])
{
    int val = 10;
    if (argc >= 2)
    {
        std::istringstream iss(argv[1]);

        if (iss >> val)
        {
            // cout << "Got cmd argument" << endl;
        }
    }
    // cout << "Using batches of " << val << endl;

    // cout << "Hello World" << endl;

    do_reads_in_batches_of(val);
}