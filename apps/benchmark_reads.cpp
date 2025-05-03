#include <iostream>
#include <cstdlib>
#include "windows_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include "utils.h"
#include "timer.h"
#include <omp.h>

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

void do_multiple_reads_with_threads(int thread_count, int batches_of)
{
    string file_name = "F:\\indices\\turing_10m\\disk_index_disk.index";
    auto reader = new WindowsAlignedFileReader();
    reader->open(file_name.c_str());

    vector<vector<AlignedRead>> read_reqs(thread_count);

    omp_set_num_threads(thread_count);

#pragma omp parallel for num_threads((int)thread_count)
    for (int i = 0; i < thread_count; i++)
    {
        reader->register_thread();
        read_reqs[i].reserve(batches_of);

        // create read requests
        for (size_t j = 0; j < batches_of; ++j)
        {
            char* buf = nullptr;
            alloc_aligned((void**)&buf, SECTOR_LEN, SECTOR_LEN);

            AlignedRead read;
            read.buf = buf;
            read.len = SECTOR_LEN;
            read_reqs[i].push_back(read);
        }
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
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < no_of_reads; i++)
    {
        do_reads(reader, read_reqs[omp_get_thread_num()], i);
    }
    // cout << "Time taken to read in microseconds: " << timer.elapsed() << endl;
    cout << timer.elapsed() << endl;

    reader->close();
}

int main(int argc, char* argv[])
{
    int thread_count = 1;
    int batches_of = 128;
    if (argc >= 2) {
        std::istringstream iss(argv[1]);
        if (iss >> thread_count)
        {
            // cout << "Got cmd argument" << endl;
        }
    }
    if (argc >= 3) {
        std::istringstream iss(argv[2]);
        if (iss >> batches_of) {
            // cout<<"Got batch size argument"<<endl;
        }
    }
    // cout << "Using " << thread_count << " threads." << endl;
    // cout << "Using batch size of " << batches_of << endl;

    // cout << "Hello World" << endl;
    do_multiple_reads_with_threads(thread_count, batches_of);
}