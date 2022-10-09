# ./build/tests/build_disk_index float l2 /data/datasets/sift1M/sift_base.bin /data/wzy/graphs/diskann/sift1M.bin \
# 32 125 3 3 32 0
./build/tests/search_disk_index float l2 /data/wzy/graphs/diskann/sift1M.bin 200 1 1 /data/datasets/sift1M/sift_query.bin /data/datasets/sift1M/sift_groundtruth.bin 10 /data/wzy/DiskANN/logs/ 10 20 30 40 50 60 70