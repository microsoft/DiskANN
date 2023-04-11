root="${HOME}/data/Yandex10M"
data_path="${root}/base.bin"
query_path="${root}/train-query_100K_data.bin"
nnid_path="${root}/results/train-query_vamana_300_idx_uint32.bin"
R=64
Lbuild=128
alpha=1.2
lambda=0.75
index_path="${root}/vamana_R${R}_L${Lbuild}_A${alpha}_f1_lambda${lambda}"

cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path --query_path $query_path --nnid_path $nnid_path \
                                    --index_path_prefix $index_path \
                                    -R $R -L $Lbuild --alpha $alpha --lambda $lambda"

# cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path \
#                                     --index_path_prefix $index_path \
#                                     -R $R -L $Lbuild --alpha $alpha"

echo $cmd
eval $cmd

K=100
query_path="${root}/dev-query.bin"
gt_path="${root}/dev-query_gt.bin"
cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

# echo $cmd
# eval $cmd

K=10
# query_path="${root}/train-query_100K_data.bin"
res_path="${root}/results/dev-query_vamana"
cmd="./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
                                -K $K -L 128 --result_path $res_path"

echo $cmd
eval $cmd