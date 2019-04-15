for K in 100; do
    for L in 64 128 256; do
	for R in 64 128 256; do
	    for C in 64 128 256; do
		echo "\n"L${L} R${R} C${C} "\n"
		./build/tests/test_nsg_index /mnt/bing_kann/qna_new/qna_base.fvecs /mnt/bing_kann/qna_new/qna_base.gr ${L} ${R} ${C} /mnt/bing_kann/qna_new/qna_base.nsg
		if [ $? -eq 0 ]
		then
		    for LS in 125 150 200; do
			echo "\n"L${L} R${R} C${C} LS${LS} "\n"
			./build/tests/test_nsg_search /mnt/bing_kann/qna_new/qna_base.fvecs /mnt/bing_kann/qna_new/qna_query.fvecs /mnt/bing_kann/qna_new/qna_base.nsg ${LS} ${K}  /mnt/bing_kann/qna_new/qna_query.res
			./build/tests/calc_recall /mnt/bing_kann/qna_new/qna_query_gs100.ivecs  /mnt/bing_kann/qna_new/qna_query.res
		    done
		else
		    echo ERROR
		fi
	    done
	done
    done
done
