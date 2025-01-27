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

void do_reads(WindowsAlignedFileReader *reader, char *buf, int batches_of)
{
    auto ctx = reader->get_ctx();

    std::vector<AlignedRead> read_reqs;
    read_reqs.reserve(batches_of);

    // create read requests
    for (size_t i = 0; i < batches_of; ++i)
    {
        AlignedRead read;
        read.len = SECTOR_LEN;
        read.buf = buf + i * SECTOR_LEN;
        auto sector_id = (rand() % 1650000);
        read.offset = sector_id * SECTOR_LEN;
        if (read.offset)
            read_reqs.push_back(read);
    }

    reader->read(read_reqs, ctx, false);
}

void do_multiple_reads_with_threads(int thread_count)
{
    string file_name = "F:\\indices\\turing_10m\\disk_index_disk.index";
    auto reader = new WindowsAlignedFileReader();
    reader->open(file_name.c_str());
    int batches_of = 100;

    vector<char *> buffers(thread_count);

    omp_set_num_threads(thread_count);

#pragma omp parallel for num_threads((int)thread_count)
    for (int i = 0; i < thread_count; i++)
    {
        char *buf = nullptr;
        alloc_aligned((void **)&buf, batches_of * SECTOR_LEN, SECTOR_LEN);
        buffers[i] = buf;
        reader->register_thread();
    }

    int no_of_reads = 10000;
    Timer timer;
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < no_of_reads; i++)
    {
        char *buf = buffers[omp_get_thread_num()];
        do_reads(reader, buf, batches_of);
    }
    // cout << "Time taken to read in microseconds: " << timer.elapsed() << endl;
    cout<<timer.elapsed()<<endl;

    reader->close();
}

int main(int argc, char *argv[])
{
    int val = 1;
    if (argc >= 2)
    {
        std::istringstream iss(argv[1]);

        if (iss >> val)
        {
            // cout << "Got cmd argument" << endl;
        }
    }
    // cout << "Using " << val << " threads." << endl;

    // cout << "Hello World" << endl;
    do_multiple_reads_with_threads(val);
}