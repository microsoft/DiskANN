import os
os.environ['OPENBLAS_NUM_THREADS'] = '24'
import array
import struct
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import math
import time
from sklearn import preprocessing # type: ignore
from sklearn.cluster import KMeans # type: ignore
import sklearn
import random
import pickle
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from itertools import combinations
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import tensorflow_hub as hub

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

INPUT_FILE_NAME = "bin/wiki_vector.bin"
LABLE_FILE_NAME = "Final/cleaned_wiki_filters.txt"
LABEL_FREQ_FILE = "label_files/label_freq.txt"
GLOBAL_N = 3535167920

# INPUT_FILE_NAME = "data/slice/wiki_vector_1m.bin"
# LABLE_FILE_NAME = "data/slice/cleaned_wiki_useful_label_1m.txt"
# LABEL_FREQ_FILE = "data/slice/label_freq_1m.txt"
# GLOBAL_N = 1000000

NO_OF_CLUSTERS = 100
KMEANS_CLUSTER_FILE_NAME = str(NO_OF_CLUSTERS) + "_kmeans_cluster.pickle"
NORMALIZED_DATA_FILE_NAME = "normalized_data.pickle"
NO_OF_THREADS = 50

CHUNK_SIZE = GLOBAL_N // NO_OF_THREADS
COSINE_SIMILARITY_THRESHOLD = 0.75
CENTROIDS_SHAPE = (200, 768)
POSTING_LIST_FILE_NAME = "posting_list_16.pickle"
CENTROID_FILE_NAME = "centroids_16.pickle"
COSINE_SIMILARITY_DEVIATION_THRESHOLD = 0.1
NUM_OF_ITER_FOR_CLUSTERING = 16

# def readData(filename, labels, label, datatype='f', datatype_size=4):
#     # Open the binary file in read-binary mode
#     with open(filename, 'rb') as file:
#         # Read the first 8 bytes and unpack them into two 4-byte integers
#         N, M = struct.unpack('ii', file.read(8))
#         print(N, M)

#         data = []
        
#         # Loop through the number of rows N
#         for _ in range(N):
#             # Initialize an empty list to store the data
#             vector = array.array(datatype)
#             # For each row, read M*datatype_size bytes and unpack them into datatype
#             vector.fromfile(file, M)
#             if label.lower().strip() in [l.lower().strip() for l in labels[_]]:
#                 data.append(vector)

#     print("Label: " + label + ", Data Length = " + str(len(data)))
#     return data

def readData(filename, labels, selected_labels, datatype='f', datatype_size=4):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        # Read the first 8 bytes and unpack them into two 4-byte integers
        N, M = struct.unpack('ii', file.read(8))
        print(N, M)

        data = {label: [] for label in selected_labels}
        
        # Loop through the number of rows N
        for i in tqdm(range(N), desc="Reading data"):
            # Initialize an empty list to store the data
            vector = array.array(datatype)
            # For each row, read M*datatype_size bytes and unpack them into datatype
            vector.fromfile(file, M)
            for label in selected_labels:
                if label.lower().strip() in [l.lower().strip() for l in labels[i]]:
                    data[label].append(vector)

    for label in data:
        print(f"Label: {label}, Data Length = {len(data[label])}")
    return data

def load_labels(label_path, num_vectors):
    with open(label_path, 'r') as f:
        labels = [line.strip().split(',') for line in f.readlines()]
    labels = labels[:num_vectors]
    return labels

def get_angular_similarity(x):
    x = min(x, 1)
    x = max(x, -1)
    return 1 - math.acos(x) / math.pi

np_getAngularSimilarity = np.vectorize(get_angular_similarity)

def dumpToFile(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

def sharedArray(data):
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
    np_data = np.ndarray(CENTROIDS_SHAPE, dtype=np.float32, buffer=shm.buf)
    np_data[:] = data[:]
    return shm, np_data

def worker(data_chunk, centroids):
    cluster_counts = np.zeros(len(centroids))
    for vector in data_chunk:
        for cluster_id in range(len(centroids)):
            similarity = cosine_similarity([vector], [centroids[cluster_id]])[0][0]
            if similarity >= COSINE_SIMILARITY_THRESHOLD:
                cluster_counts[cluster_id] += 1
    return cluster_counts

def getLabelDistributionPerCluster(data, centroids):
    chunk_size = len(data) // NO_OF_THREADS
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with multiprocessing.Pool(processes=NO_OF_THREADS) as pool:
        results = pool.starmap(worker, [(chunk, centroids) for chunk in data_chunks])
    
    cluster_counts = np.sum(results, axis=0)
    return cluster_counts

def select_labels_by_frequency(label_frequencies, ranges):
    selected_labels = []
    for min_freq, max_freq, num_labels in ranges:
        labels_in_range = [label for label, freq in label_frequencies.items() if min_freq <= freq < max_freq]
        np.random.shuffle(labels_in_range)
        selected_labels.extend(labels_in_range[:num_labels])
    return selected_labels

def select_labels(LABEL_FREQ_FILE, threshold):
    frequency_ranges = [
        # (400000, 600000, 2),
        # (200000, 400000, 4),
        (100000, 200000, 6),
        (50000, 100000, 8),
        (10000, 50000, 10),
        (1000, 10000, 10),
        (200, 1000, 10)
    ]
    
    label_frequencies = {}
    with open(LABEL_FREQ_FILE, 'r') as f:
        for line in f:
            label, freq = line.strip().split(':')
            label_frequencies[label.strip()] = int(freq.strip())
    
    selected_labels = select_labels_by_frequency(label_frequencies, frequency_ranges)
    return selected_labels

if __name__ == "__main__":
    start_time = time.time()
    
    labels = load_labels(LABLE_FILE_NAME, GLOBAL_N)
    selected_labels = select_labels(LABEL_FREQ_FILE, 1000)
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Universal Sentence Encoder loaded.")
    
    label_embeddings = embed(selected_labels)
    label_embeddings = np.array(label_embeddings)
    print("Label embeddings computed.")
    
    label_pairs = set(combinations(selected_labels, 2))    
                
    label_pair_similarities = []
    similarity_dict = {}
    for label1, label2 in label_pairs:
        sim = np.dot(label_embeddings[selected_labels.index(label1)], label_embeddings[selected_labels.index(label2)]) / (np.linalg.norm(label_embeddings[selected_labels.index(label1)]) * np.linalg.norm(label_embeddings[selected_labels.index(label2)]))
        label_pair_similarities.append((label1, label2, sim))
        similarity_dict[(label1, label2)] = sim
    print("Label pair similarities computed.")
    
    #take top 5 and bottom 5 similar pairs
    similarity_dict = dict(sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)[:10] + sorted(similarity_dict.items(), key=lambda x: x[1], reverse=False)[:10])
    # take random 20 pairs
    # similarity_dict = dict(random.sample(similarity_dict.items(), 20))
    print("Selected label pairs based on similarity. Size = " + str(len(similarity_dict)))
        
    data = {}
    final_labels = []
    for label1, label2 in similarity_dict:
        final_labels.append(label1)
        final_labels.append(label2)
    final_labels = list(set(final_labels))
    
    data = readData(INPUT_FILE_NAME, labels, final_labels)

    print("Data loaded for all selected labels.")
    
    for label in data:
        data[label] = preprocessing.normalize(data[label])
    print("Data Normalized")
    
    pair_centroids = []
    
    for pair in similarity_dict:
        label1, label2 = pair
        union = np.vstack((data[label1], data[label2]))
        kmeans = KMeans(n_clusters=NO_OF_CLUSTERS, random_state=0).fit(union)
        centroids = kmeans.cluster_centers_
        pair_centroids.append((label1, label2, centroids))
    
    print("KMeans Clustering Done")
    
    label_distribution = []
    metrics = []
    
    # Create the directory for saving plots
    output_dir = "analysis_result/label_pair_distribution_0312_large"
    os.makedirs(output_dir, exist_ok=True)

    for label1, label2, pair_centroid in tqdm(pair_centroids, desc="Label pairs processed"):
        label1_distribution = getLabelDistributionPerCluster(data[label1], pair_centroid)
        label2_distribution = getLabelDistributionPerCluster(data[label2], pair_centroid)
        
        # Normalize the counts
        label1_total = len(data[label1])
        label2_total = len(data[label2])
        label1_distribution = label1_distribution / label1_total
        label2_distribution = label2_distribution / label2_total
        
        distribution = [(i, label1_distribution[i], label2_distribution[i]) for i in range(NO_OF_CLUSTERS)]
        label_distribution.append((label1, label2, distribution))
    first = 0
    for label1, label2, distribution in label_distribution:
        print(f"Label distribution for pair ({label1}, {label2}):")
        # for cluster_id, label1_count, label2_count in distribution:
            # print(f"Cluster {cluster_id}: {label1} count = {label1_count}, {label2} count = {label2_count}")
            
        # Calculate the percentage of label1 in each cluster
        percentages = [label1_count / (label1_count + label2_count) if (label1_count + label2_count) != 0 else 0 for _, label1_count, label2_count in distribution]
        if (first == 0):
            bins = np.arange(0, 60, 10) / 100
            bucketed_percentages = np.digitize(percentages, bins) - 1
            print(bucketed_percentages)
        # convert percetages between 50 to 100, to 100 - percentages
        for i in range(len(percentages)):
            if percentages[i] > 0.5:
                percentages[i] = 1 - percentages[i]
                
        if (first == 0):
            bins = np.arange(0, 60, 10) / 100
            bucketed_percentages = np.digitize(percentages, bins) - 1
            print(bucketed_percentages)
            first = 1
        
        # Calculate metrics
        mean_percentage = np.mean(percentages)
        # median_percentage = np.median(percentages)
        # std_percentage = np.std(percentages)
        similarity = similarity_dict[(label1, label2)]
        
        metrics.append((similarity, mean_percentage))
        
        # Plot the distribution of bucketed percentages
        bins = np.arange(0, 60, 10) / 100
        bucketed_percentages = np.digitize(percentages, bins) - 1
        
        plt.figure(figsize=(10, 6))
        plt.hist(bucketed_percentages, bins=len(bins)-1, alpha=0.6, edgecolor='black')
        plt.xticks(ticks=np.arange(len(bins)-1), labels=[f'{int(b*100)}-{int((b+0.1)*100)}%' for b in bins[:-1]])
        plt.xlabel(f'Percentage of {label1} in cluster')
        plt.ylabel('Number of Clusters')
        plt.title(f'Percentage Distribution of {label1}, pair ({label1[:10]}, {label2[:10]}), [Sim = {similarity:.4f}]')
        plt.text(0.95, 0.95, f'Num vectors: {len(data[label1])}, {len(data[label2])}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        # plt.show()
        plt.savefig(f"{output_dir}/{label1}_{label2}_distribution.png")
        plt.close()
        
    # Plot metrics against similarity
    similarities, means = zip(*metrics)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(similarities, means, label='Mean', color='blue')
    # plt.scatter(similarities, medians, label='Median', color='green')
    # plt.scatter(similarities, stds, label='Standard Deviation', color='red')
    plt.xlabel('Similarity')
    plt.ylabel('Metric Value')
    plt.title('Metrics of Label1 Percentage Distribution vs Similarity')
    plt.legend()
    plt.savefig(f"{output_dir}/metrics_vs_similarity.png")
    plt.show()
    
    # Save the results label-pair, similarity, and metrics in a file
    with open(f"{output_dir}/label_pair_similarity.txt", 'w') as f:
        f.write("Label1, Label2, Similarity, Mean, Num_vectors1, Num_vectors2\n")
        for (l1, l2), sim in similarity_dict.items():
            mean = next(m for s, m in metrics if s == sim)
            # median = next(med for s, _, med, _ in metrics if s == sim)
            # std = next(st for s, _, _, st in metrics if s == sim)
            num_vectors1 = len(data[l1])
            num_vectors2 = len(data[l2])
            f.write(f"{l1}, {l2}, {sim:.4f}, {mean:.4f}, {num_vectors1}, {num_vectors2}\n")
    
    elapsed_time = time.time() - start_time
    print("Time Taken = " + str(elapsed_time))