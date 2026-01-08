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

vector<long long int> random_sector_ids(TOTAL_READS);

void do_reads(WindowsAlignedFileReader* reader, vector<AlignedRead>& read_reqs, int batch_num)
{
    auto ctx = reader->get_ctx();
    size_t batch_size = read_reqs.size();

    // Modify read requests
    for (int i = 0; i < batch_size; i++)
    {
        read_reqs[i].offset = random_sector_ids[batch_num * batch_size + i] * SECTOR_LEN;
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
    for (int i = 0; i < TOTAL_READS; i++)
    {
        random_sector_ids[i] = distrib(gen);
    }

    int no_of_reads = TOTAL_READS / batches_of;
    Timer timer;
    for (int i = 0; i < no_of_reads; i++)
    {
        do_reads(reader, read_reqs, i);
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