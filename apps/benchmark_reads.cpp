#include <iostream>
#include <cstdlib>
#include "windows_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include "utils.h"
#include "timer.h"
#include <omp.h>
#include <concurrent_queue.h>

using namespace std;
using namespace diskann;

#define SECTOR_LEN 4096

void do_reads(WindowsAlignedFileReader *reader, char *buf)
{
    auto ctx = reader->get_ctx();

    std::vector<AlignedRead> read_reqs;
    int num_sectors = 100;

    // create read requests
    for (size_t i = 0; i < num_sectors; ++i)
    {
        AlignedRead read;
        read.len = SECTOR_LEN;
        read.buf = buf + i * SECTOR_LEN;
        auto sector_id = (rand() % 10000);
        read.offset = sector_id * SECTOR_LEN;
        if (read.offset)
            read_reqs.push_back(read);
    }

    reader->read(read_reqs, ctx, false);
}

void do_multiple_reads_with_threads(int thread_count)
{
    string file_name = "C:\\DiskANN\\Data\\turning_100k\\index_disk.index";
    auto reader = new WindowsAlignedFileReader();
    reader->open(file_name.c_str());
    int num_sectors = 100;

    ConcurrentQueue<char*> buffer_pool;

    omp_set_num_threads(thread_count);

#pragma omp parallel for num_threads((int)thread_count)
    for (int i = 0; i < thread_count; i++)
    {
        char *buf = nullptr;
        alloc_aligned((void **)&buf, num_sectors * SECTOR_LEN, SECTOR_LEN);
        buffer_pool.push(buf);
        reader->register_thread();
    }

    Timer timer;
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < 10000; i++)
    {
        char *buf = buffer_pool.pop();
        do_reads(reader, buf);
        buffer_pool.push(buf);
    }
    cout << "Time taken to read in microseconds: " << timer.elapsed() << endl;

    reader->close();
}

int main(int argc, char *argv[])
{
    int val = 1;
    if (argc >= 2)
    {
        std::istringstream iss( argv[1] );

        if (iss >> val)
        {
            cout<<"Got cmd argument"<<endl;
        }
    }
    cout<<"Using "<<val<<" threads."<<endl;

    cout << "Hello World" << endl;
    do_multiple_reads_with_threads(val);
}