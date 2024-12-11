#include <iostream>
#include <cstdlib>
#include "windows_aligned_file_reader.h"
#include "aligned_file_reader.h"
#include "utils.h"

using namespace std;
using namespace diskann;

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

#define SECTOR_LEN 4096

void do_reads()
{
    string file_name = "C:\\DiskANN\\Data\\turning_100k\\index_disk.index";
    auto reader = new WindowsAlignedFileReader();
    reader->open(file_name.c_str());
    auto ctx = reader->get_ctx();

    std::vector<AlignedRead> read_reqs;
    int num_sectors = 100;

    char *buf = nullptr;
    alloc_aligned((void **)&buf, num_sectors * SECTOR_LEN, SECTOR_LEN);

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

    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++)
    {
        reader->read(read_reqs, ctx, false);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    cout << "Time taken to read: " << diff.count() << endl;
}

int main()
{
    cout << "Hello World" << endl;
    do_reads();
}