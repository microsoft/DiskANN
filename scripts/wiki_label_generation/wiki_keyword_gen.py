"""Generate keywords and labels for Wikipedia documents.
More info: https://microsoftapc-my.sharepoint.com/:w:/g/personal/t-asutradhar_microsoft_com/Ec7UTAvWVhpKvBiDSF8fLIcBLtwTdKzqb7_L-lD2HSNsGw?e=TeLmgB"""

import os

# Set TensorFlow optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


import time
import multiprocessing
from multiprocessing import Pool
import yake
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datasets import load_dataset, Dataset

"""constants"""
num_keywords = 100
max_no_labels = 20
min_no_labels = 10
similarity_threshold = 0.4
num_processes = 56


### -------------------------------------- ###
###         Step 1: Keyword Extraction     ###
### -------------------------------------- ###
kw_extractor = yake.KeywordExtractor(top=num_keywords, n=2)

def extract_keywords(text, max_keywords=20):
    """Extract keywords from text using YAKE."""
    return [kw for kw, _ in kw_extractor.extract_keywords(text)]

def generate_keywords(wiki_data):
    """Multiprocess keyword extraction."""
    start_time = time.time()

    with multiprocessing.Pool(num_processes) as pool:
        wiki_data_texts = wiki_data['text']
        keywords_list = list(tqdm(pool.imap(extract_keywords, wiki_data_texts), total=len(wiki_data_texts), desc="Extracting keywords"))

    wiki_data = wiki_data.add_column('keywords', [", ".join(kw) for kw in keywords_list])
    print(f"[{time.strftime('%H:%M:%S')}] Keyword extraction completed in {time.time() - start_time:.2f} sec.")
    
    return wiki_data

### -------------------------------------- ###
###         Step 2: Label Generation       ###
### -------------------------------------- ###

def embed_text(texts, embed):
    """Generate embeddings for text using Universal Sentence Encoder."""
    return embed(texts).numpy()

def process_chunk(chunk, embed_url):
    """Process a chunk of documents and generate labels."""
    embed = hub.load(embed_url)  # Load the model in each process

    all_labels = []
    
    chunk = chunk.to_pandas().to_dict('records')  # Convert to a list of dictionaries
    
    for doc in chunk:
        document_text = doc['text']
        keywords = doc['keywords'].split(', ')  # Convert to list
        title = doc['title']
        document_embedding = embed_text([document_text], embed)[0]  # Get document embedding

        # Generate keyword embeddings
        keyword_embeddings = embed_text(keywords, embed)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(document_embedding.reshape(1, -1), keyword_embeddings).flatten()

        # Get top relevant keywords
        top_indices = np.where(cosine_similarities >= similarity_threshold)[0]
        if len(top_indices) < min_no_labels:
            top_indices = np.argsort(cosine_similarities)[-(min_no_labels):]  # Get top 10 if less than 10
        top_labels = [keywords[idx] for idx in top_indices[:(max_no_labels)]]  # Max 20 labels
        top_labels.append(title)  # Add the title as a label
        #remove duplicates
        top_labels = list(set(top_labels))

        all_labels.append(", ".join(top_labels))
    
    return all_labels


def generate_labels(keyword_data):
    """Assign labels to documents based on keyword embeddings."""
    start_time = time.time()
    
    # Split the keyword data into chunks for multiprocessing
    chunk_size = len(keyword_data) // num_processes
    chunks = [keyword_data.select(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_processes)]
    
    embed_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # The model URL
    
    # Use multiprocessing to process chunks in parallel
    with multiprocessing.Pool(num_processes) as pool:
        with tqdm(total=len(chunks), desc="Selecting labels") as pbar:
            results = []
            for result in pool.starmap(process_chunk, [(chunk, embed_url) for chunk in chunks]):
                results.append(result)
                pbar.update(1)
    
    # Flatten the list of results and add the labels to the original data
    # for chunk in chunks:
    #     chunk_titles = chunk['title']
    #     for idx, title in enumerate(chunk_titles):
    #         if idx < len(results):
    #             results[idx].append(title)
    #         else:
    #             print(f"Index {idx} out of range for results with length {len(results)}")

    all_labels = [label for sublist in results for label in sublist]
    # Ensure the lengths match before adding the column
    if len(all_labels) == len(keyword_data):
        keyword_data = keyword_data.add_column('labels', all_labels)
    else:
        print(f"Length mismatch: all_labels has {len(all_labels)} items, but keyword_data has {len(keyword_data)} rows")

    print(f"[{time.strftime('%H:%M:%S')}] Label generation completed in {time.time() - start_time:.2f} sec.")
    
    return keyword_data

### -------------------------------------- ###
###         Step 3: Pipeline Execution     ###
### -------------------------------------- ###
def main():
    current_date_time = time.strftime('%Y%m%d%H%M%S')
    # keywords_path = f'results/keywords_{current_date_time}.arrow'
    # final_labels_path = f'results/final_labels_{current_date_time}.arrow'

    print(f"[{time.strftime('%H:%M:%S')}] Loading data...")
    # Load data
    start_time_1 = time.time()

    dataset = load_dataset('Cohere/wikipedia-22-12-en-embeddings')
    wiki_data = dataset['train']
    # wiki_data = wiki_data.select(range(1120))  # Limit for testing
    print("Size of wiki_data is: ", len(wiki_data))

    print(f"[{time.strftime('%H:%M:%S')}] Data loaded in {time.time() - start_time_1:.2f} sec.")
    print(f"[{time.strftime('%H:%M:%S')}] Data loaded. Processing...")

    data_chunk_size = 628040
    num_chunks = len(wiki_data) // data_chunk_size + (1 if len(wiki_data) % data_chunk_size != 0 else 0)

    for i in range(num_chunks):
        chunk_start = i * data_chunk_size
        chunk_end = min((i + 1) * data_chunk_size, len(wiki_data))
        chunk = wiki_data.select(range(chunk_start, chunk_end))

        print(f"[{time.strftime('%H:%M:%S')}] Processing chunk {i + 1}/{num_chunks}...")

        # Step 1: Generate keywords for the chunk

        keyword_data_chunk = generate_keywords(chunk)
        chunk_keywords_path = f'results/chunk_{i + 1}_keywords_{current_date_time}.arrow'
        keyword_data_chunk.save_to_disk(chunk_keywords_path)
        print(f"[{time.strftime('%H:%M:%S')}] Keywords for chunk {i + 1} saved.")

        # Step 2: Generate labels for the chunk
        labeled_data_chunk = generate_labels(keyword_data_chunk)
        chunk_labels_path = f'results/chunk_{i + 1}_labels_{current_date_time}.arrow'
        labeled_data_chunk.save_to_disk(chunk_labels_path)
        print(f"[{time.strftime('%H:%M:%S')}] Labels for chunk {i + 1} saved.")
    
    print(f"[{time.strftime('%H:%M:%S')}] Processing complete in {time.time() - start_time_1:.2f} sec. Results saved.")

if __name__ == "__main__":
    main()