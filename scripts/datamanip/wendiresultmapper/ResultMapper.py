import sys
import pandas as pd 
import linecache

def print_one(table_name, df):
    print(f"Table: {table_name}")
    header = pd.DataFrame([df.columns], columns=df.columns)  # Create a row with headers
    first_row = df.iloc[[0]]  # Get the first row
    result = pd.concat([header, first_row])
    print(result.to_string(index=False))
    print("\n\n")

def main():

    qhash_id_map_file = "E:\\data\\FromWendi\\new\\query_hash_vecid_map.tsv"
    query_sf_filters_file = "E:\\data\\FromWendi\\new\\query_filters.tsv"
    dhash_docid_map_file = "E:\\data\\FromWendi\\new\\doc_hash_vecid_map.tsv"
    query_results_raw_file = "D:\\bin\\github\\new-mainbranch-wendi-rslts_L0_results.tsv"

    final_results_file = "E:\\data\\FromWendi\\new\\final_results_file.tsv"
    

    cn1 = ['query_hash', 'query_vec_id']
    qhash_id_map_df = pd.read_csv(qhash_id_map_file, sep='\t', header=None, names=cn1, encoding='utf-8')
    print_one("QueryHash To QueryId", qhash_id_map_df)

    cn2 = ['filter']
    query_sf_filters_df = pd.read_csv(query_sf_filters_file, sep='\t', header=None, names=cn2, encoding='utf-8')
    query_sf_filters_df.reset_index(drop=False, inplace=True)
    query_sf_filters_df.rename(columns={'index': 'query_vec_id'}, inplace=True)
    print_one("Query Filters", query_sf_filters_df)

    cn3 = ['doc_hash', 'doc_vec_id']
    dhash_docid_map_df = pd.read_csv(dhash_docid_map_file, sep='\t', header=None, names=cn3, encoding='utf-8')
    print_one("DocHash To DocId", dhash_docid_map_df)

    cn4 = ['query_vec_id', 'results']
    query_results_raw_df = pd.read_csv(query_results_raw_file, sep='\t', header=None, names=cn4, encoding='utf-8')
    print_one("Query Results Raw", query_results_raw_df)

    cn5 = ['query_vec_id', 'doc_vec_id', 'score', 'match_type']
    processed_results_df = pd.DataFrame(columns=cn5)

    for index, row in query_results_raw_df.iterrows():
        result_str = row['results']
        detailed_result_list = result_str.split('),')
        if index % 1000 == 0:
            print("Processing row: {}".format(index))
        detailed_result_rows = {'query_vec_id': [], 'doc_vec_id': [], 'score': [], 'match_type': []}
        for detailed_result in detailed_result_list:
            detailed_result = detailed_result.strip('(').strip(')').strip()
            if detailed_result == '':
                continue
            result_id_score_match = detailed_result.split(',')
            detailed_result_rows['query_vec_id'].append(row['query_vec_id'])
            detailed_result_rows['doc_vec_id'].append(result_id_score_match[0])
            detailed_result_rows['score'].append(result_id_score_match[1])
            detailed_result_rows['match_type'].append(result_id_score_match[2])
        processed_results_df = pd.concat([processed_results_df, pd.DataFrame(detailed_result_rows)], ignore_index=True)
    print_one("Processed Results", processed_results_df)

    #If there is a possibility of running out of memory while processing this data
    #save the processed_results_df to a file and read it back.
    #processed_results_df.to_csv("E:\\data\\FromWendi\\new\\results_with_query_and_docids.tsv", sep='\t', index=False)
    #Do the final merge between processed_results_df and dhash_docid_map_df
    # cn5 = ['query_vec_id', 'doc_vec_id', 'score', 'match_type']
    # processed_results_df = pd.read_csv("E:\\data\\FromWendi\\new\\results_with_query_and_docids.tsv", names=cn5, sep='\t', encoding='utf-8')
    # print_one("Processed Results", processed_results_df)

    processed_results_with_filters = pd.merge(processed_results_df, query_sf_filters_df, on = 'query_vec_id', how='inner')
    print_one("Results With Filters", processed_results_with_filters)

    results_with_query_hash = pd.merge(processed_results_with_filters, qhash_id_map_df, on = 'query_vec_id', how='inner')
    final_results = pd.merge(results_with_query_hash, dhash_docid_map_df, on = 'doc_vec_id', how='inner')
    final_results.to_csv(final_results_file, sep='\t', index=False)


    

    # # qhash, qid, filter
    # qhash_qid_filter_df = pd.merge(qhash_id_map_df, query_sf_filters_df, on = 'query_vec_id', how='inner')
    # # qhash, qid, filter, docid, score, matchtype 
    # qhash_qid_filter_docid_score_match_df = pd.merge(qhash_qid_filter_df, processed_results_df, on = 'query_vec_id', how='inner')
    # # qhash, qid, filter, docid, score, matchtype, dochash
    # qhash_qid_filter_docid_score_match_dochash_df = pd.merge(qhash_qid_filter_docid_score_match_df, dhash_docid_map_df, on = 'doc_vec_id', how='inner')
    # qhash_qid_filter_docid_score_match_dochash_df.to_csv(final_results_file, sep='\t', index=False)

    
    





    

    # cn1 = ['incoming_query_id', 'query_vec', 'labels']
    # origquery_df = pd.read_csv(origquery_file, sep='\t', header=None, names=cn1, encoding='utf-8')

    # cn2=["query_vec"]
    # query_pipe_sep_df = pd.read_csv(query_pipe_sep_file, sep='\t', header=None, names=cn2, encoding='utf-8')
    # query_pipe_sep_df.reset_index(drop=False, inplace=True)
    # query_pipe_sep_df.rename(columns={'index': 'query_id'}, inplace=True)

    # cn3=["label"]
    # query_sf_df = pd.read_csv(query_sf_file, sep='\t', header=None, names=cn3, encoding='utf-8')
    # query_sf_df.reset_index(drop=False, inplace=True)
    # query_sf_df.rename(columns={'index': 'query_id'}, inplace=True)

    # cn4=["query_id", "results"]
    # results_df = pd.read_csv(results_file, sep='\t', header=None, names=cn4, encoding='utf-8')
    # results_df.reset_index(drop=False)

    # #print column names of each dataframe
    # print("Columns of ORIGINAL query file:{}".format(origquery_df.columns))
    # print("Columns of PROCESSED query file: {}".format(query_pipe_sep_df.columns))
    # print("Columns of QUERY FILTERS file: {}".format(query_sf_df.columns))
    # print("Columns of RESULTS file: {}".format(results_df.columns))
    

    # #merge the dataframes carefuly!
    # query_tsv_orig_query_joined = pd.merge(origquery_df, query_pipe_sep_df, on = 'query_vec', how='inner')
    # incoming_q_id_query_id = query_tsv_orig_query_joined[['query_id', 'incoming_query_id']]
    # incoming_q_id_query_id_with_labels = pd.merge(incoming_q_id_query_id, query_sf_df, on = 'query_id', how='inner')
    # incoming_q_id_query_id_with_labels_results = pd.merge(incoming_q_id_query_id_with_labels, results_df, on = 'query_id', how='inner')

    # print("Merged ORIGINAL query file WITH single filters file and results and obtained {} rows and these columns:{}".format(incoming_q_id_query_id_with_labels_results.shape[0], incoming_q_id_query_id_with_labels_results.columns))
    # print("Now processing the results to get the doc_ids, scores, and match types.")

    # final_result_list = pd.DataFrame(columns=['incoming_query_id', 'label', 'doc_id', 'score', 'match_type'])
    # #loop through the dataframes
    # for index, row in incoming_q_id_query_id_with_labels_results.iterrows():
    #     result_str = row['results']
    #     detailed_result_list = result_str.split('),')
    #     print("Process row: {} with query_id:{} and label: {}".format(index, row['incoming_query_id'], row['label']))
    #     detailed_result_rows = {'incoming_query_id': [], 'label': [], 'doc_id': [], 'score': [], 'match_type': []}
    #     for detailed_result in detailed_result_list:
    #         detailed_result = detailed_result.strip('(').strip(')').strip()
    #         if detailed_result == '':
    #             continue
    #         result_id_score_match = detailed_result.split(',')
    #         #new_record = pd.DataFrame([{'incoming_query_id': row['incoming_query_id'], 'label': row['label'], 'doc_id': result_id_score_match[0], 'score': result_id_score_match[1], 'match_type': result_id_score_match[2]}])
    #         detailed_result_rows['incoming_query_id'].append(row['incoming_query_id'])
    #         detailed_result_rows['label'].append(row['label'])
    #         detailed_result_rows['doc_id'].append(result_id_score_match[0])
    #         detailed_result_rows['score'].append(result_id_score_match[1])
    #         detailed_result_rows['match_type'].append(result_id_score_match[2])
    #     final_result_list = pd.concat([final_result_list, pd.DataFrame(detailed_result_rows)], ignore_index=True)

    # final_result_list.to_csv(results_with_doc_ids_file, sep='\t', index=False)        
    # print("Obtained {} records after extracting results into separate rows. Saved to file {}.".format(final_result_list.shape[0], results_with_doc_ids_file))

    # final_result_list = pd.read_csv(results_with_doc_ids_file, sep='\t', encoding='utf-8')

    # print("Reading docs master from: {}".format(docs_master_file))
    # docs_master = pd.read_csv(docs_master_file, sep='\t', usecols=[0], names=['doc_hash'], encoding='utf-8')
    # docs_master.reset_index(drop=False, inplace=True)
    # docs_master.rename(columns={'index': 'doc_id'}, inplace=True)
    # print("Docs master has {} rows and these columns:{}".format(docs_master.shape[0], docs_master.columns))

    # print("Merging final result set with docs_master with {} rows and these columns {}".format(docs_master.shape[0], docs_master.columns))
    # final_result_with_doc_hashes = pd.merge(final_result_list, docs_master, on = 'doc_id', how='inner')
    # final_result_with_doc_hashes.to_csv("E:\\data\\FromWendi\\results_with_doc_hashes.tsv", sep='\t', index=False)

if __name__ == '__main__':
    main()
