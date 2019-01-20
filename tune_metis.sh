for K in 100; do
    for L in 2000; do
	for R in 32 64 128 256; do
	    for C in 10000; do
		echo "\n"L${L} R${R} C${C} "\n"
		./build/tests/test_nsg_index /mnt/bing_kann/metis_new/metis_base_new.fvecs /mnt/bing_kann/metis_new/metis_base_new.gr ${L} ${R} ${C} /mnt/bing_kann/metis_new/metis_base_new.nsg
		if [ $? -eq 0 ]
		then
		    for LS in 500; do
			echo "\n"L${L} R${R} C${C} LS${LS} "\n"
			./build/tests/test_nsg_search /mnt/bing_kann/metis_new/metis_base_new.fvecs /mnt/bing_kann/metis_new/metis_query.fvecs /mnt/bing_kann/metis_new/metis_base_new.nsg ${LS} ${K} /mnt/bing_kann/metis_new/metis_query.res
			./build/tests/calc_recall /mnt/bing_kann/metis_new/metis_query_gs100.ivecs  /mnt/bing_kann/metis_new/metis_query.res
		    done
		else
		    echo ERROR
		fi
	    done
	done
    done
done
