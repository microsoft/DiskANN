cd build
./apps/search_memory_index --data_type int8 --dist_fn l2 --index_path_prefix /nvmessd1/fbv4/avarhade/prec40M_sorted_memory_index_r64_l100 --gt_file /nvmessd1/fbv4/gt100_prec40M --query_file /nvmessd1/fbv4/queries384d.bin  --result_path /home/rakri/avarhade/Dump/tmp -K 50 -L 50 60 70 80 90 100 -T 1 > /home/rakri/avarhade/DiskANN/prec40M/sorted_r64_k50.txt

