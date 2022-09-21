# ./build/tests/build_disk_index float l2 /home/ubuntu/dataset/sift/sift_base_with_header.fvecs /home/ubuntu/graphs/diskann/sift1M.bin \
# 32 125 3 3 1 0
./build/tests/search_disk_index float l2 /home/ubuntu/graphs/diskann/sift1M.bin \
10 1 1 /home/ubuntu/dataset/sift/sift_query_with_header.fvecs \
/home/ubuntu/dataset/sift/sift_gt_with_header.ivecs 10 /home/ubuntu/DiskANN/python/logs/diskann 1 \
-L 50