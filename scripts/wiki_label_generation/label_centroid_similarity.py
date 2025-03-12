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
import os

os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['NUMEXPR_NUM_THREADS'] = '24'


# INPUT_FILE_NAME = "data\slice\wiki_vector_1m.bin"
# LABLE_FILE_NAME = "data\slice\cleaned_wiki_label_1m.txt"
# GLOBAL_N = 1000000

INPUT_FILE_NAME = "bin\wiki_vector.bin"
LABLE_FILE_NAME = "Final\cleaned_wiki_filters.txt"
GLOBAL_N = 3535167920

NO_OF_CLUSTERS = 200
KMEANS_CLUSTER_FILE_NAME = str(NO_OF_CLUSTERS) + "_kmeans_cluster.pickle"
NORMALIZED_DATA_FILE_NAME = "normalized_data.pickle"
NO_OF_THREADS = 24

CHUNK_SIZE = GLOBAL_N//NO_OF_THREADS
COSINE_SIMILARITY_THRESHOLD = 0.75
CENTROIDS_SHAPE = (200, 768)
POSTING_LIST_FILE_NAME = "posting_list_16.pickle"
CENTROID_FILE_NAME = "centroids_16.pickle"
COSINE_SIMILARITY_DEVIATION_THRESHOLD = 0.1 
NUM_OF_ITER_FOR_CLUSTERING = 16


def readData(filename, labels, label, datatype='f', datatype_size=4):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        # Read the first 8 bytes and unpack them into two 4-byte integers
        N, M = struct.unpack('ii', file.read(8))
        print(N, M)

        # randomSampleSpace = set(random.sample(range(N), 1000000))
    
        data = []
        
        # Loop through the number of rows N
        for _ in range(N):
            # Initialize an empty list to store the data
            vector = array.array(datatype)
            # For each row, read M*datatype_size bytes and unpack them into datatype
            vector.fromfile(file, M)
            # if _ in randomSampleSpace:
            #     data.append(vector)
            if label.lower().strip() in [l.lower().strip() for l in labels[_]]:
                data.append(vector)

            # if(_%1000000==0):
            # print(str(_) + " done")
            # print("Data Length = " + str(len(data)))

    print("Data Length = " + str(len(data)))
    return data

def load_labels(label_path, num_vectors):
    with open(label_path, 'r') as f:
        labels = [line.strip().split(',') for line in f.readlines()]
    labels = labels[:num_vectors]
    return labels

# Function to covert the similarity to angular similarity
def get_angular_similarity(x):
    x = min(x, 1)
    x = max(x, -1)
    return 1 - math.acos(x) / math.pi


# Vectorized version
np_getAngularSimilarity = np.vectorize(get_angular_similarity)


# Returns the labels and the number of clusters
def clusterData(data, threshold):
    dataframe = pd.DataFrame(data).to_numpy()
    N = len(dataframe)

    # Size of each bucket
    bucket_cnt = np.array([1])

    # Sum of all vectors in each bucket
    bucket_sum = np.array([dataframe[0]])

    # Label for each vector
    labels = [0]

    for i in range(1, N):
        # No. of buckets
        no_of_buckets = len(bucket_cnt)

        # Calculating Angular Similarity btw vector[i] and centroid of each bucket
        similarity_with_bucket = np_getAngularSimilarity(cosine_similarity([dataframe[i]], [np.divide(bucket_sum[i],bucket_cnt[i]) for i in range(no_of_buckets)]))[0]

        # Getting Index of bucket which is giving highest similarity
        max_idx = np.argmax(similarity_with_bucket)
        max_val = similarity_with_bucket[max_idx]

        # Check on Threshold
        if max_val >= threshold:
            # Label the utter and increase the size of bucket
            labels.append(max_idx)
            bucket_cnt[max_idx] += 1
            bucket_sum[max_idx] = np.add(bucket_sum[max_idx], dataframe[i])
        else:
            # Create New Bucket and update required variables
            new_bucket_no = len(bucket_cnt)
            labels.append(new_bucket_no)
            bucket_cnt = np.append(bucket_cnt, 1)
            bucket_sum = np.vstack((bucket_sum, dataframe[i]))
    

    # Re-assign the vectors to the buckets
    for i in range(N):
        # Calculating Angular Similarity btw vector[i] and centroid of each bucket
        similarity_with_bucket = np_getAngularSimilarity(cosine_similarity([dataframe[i]], [np.divide(bucket_sum[i],bucket_cnt[i]) for i in range(no_of_buckets)]))[0]
        
        # Getting Index of bucket which is giving highest similarity
        max_idx = np.argmax(similarity_with_bucket)
        max_val = similarity_with_bucket[max_idx]

        labels[i] = max_idx


    return labels, len(bucket_cnt)


def dumpToFile(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sharedArray(data):
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
    np_data = np.ndarray(CENTROIDS_SHAPE, dtype=np.float32, buffer=shm.buf)
    np_data[:] = data[:]
    return shm, np_data

# Function to covert the similarity to angular similarity
def get_angular_similarity(x):
    x = min(x, 1)
    x = max(x, -1)
    return 1 - math.acos(x) / math.pi


# Vectorized version
np_getAngularSimilarity = np.vectorize(get_angular_similarity)

def dumpToFile(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadFromFile(filename):
    result = None
    with open(filename, 'rb') as handle:
        result = pickle.load(handle)
    return result

def getSimilarityDistribution(data, offset, shr_name, output_queue):
    print("offset = " + str(offset) + " started", flush=True)
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shr_name)
    centroids = np.ndarray(CENTROIDS_SHAPE, dtype=np.float32, buffer=existing_shm.buf)

    similarity = np_getAngularSimilarity(cosine_similarity(data, centroids))
    similarity_above_90 = 0
    similarity_above_85 = 0
    similarity_above_80 = 0
    similarity_above_75 = 0
    similarity_above_70 = 0
    similarity_above_65 = 0
    similarity_above_60 = 0
    similarity_above_55 = 0
    similarity_above_50 = 0
    similarity_above_40 = 0
    similarity_above_30 = 0
    similarity_above_20 = 0
    similarity_above_10 = 0
    similarity_below_10 = 0

    for i in range(len(data)):
        # Getting Index of bucket which is giving highest similarity
        max_idx = np.argmax(similarity[i])
        max_val = similarity[i][max_idx]
        if max_val > 0.9:
            similarity_above_90 += 1
        elif max_val > 0.85:
            similarity_above_85 += 1
        elif max_val > 0.8:
            similarity_above_80 += 1
        elif max_val > 0.75:
            similarity_above_75 += 1
        elif max_val > 0.7:
            similarity_above_70 += 1
        elif max_val > 0.65:
            similarity_above_65 += 1
        elif max_val > 0.6:
            similarity_above_60 += 1
        elif max_val > 0.55:
            similarity_above_55 += 1
        elif max_val > 0.5:
            similarity_above_50 += 1
        elif max_val > 0.4:
            similarity_above_40 += 1
        elif max_val > 0.3:
            similarity_above_30 += 1
        elif max_val > 0.2:
            similarity_above_20 += 1
        elif max_val > 0.1:
            similarity_above_10 += 1
        else:
            similarity_below_10 += 1

    result = [similarity_above_90, similarity_above_85, similarity_above_80, similarity_above_75, similarity_above_70, similarity_above_65, similarity_above_60, similarity_above_55, similarity_above_50, similarity_above_40, similarity_above_30, similarity_above_20, similarity_above_10 ,similarity_below_10]
    # print(result, flush = True)
    output_queue.put(result)
    print("offset = " + str(offset) + " done", flush=True)


def printSimilarityDistributionWhileProcessingDataInChunks(data, centroids, label):
    pool = multiprocessing.Pool(processes=61)
    output_queue = multiprocessing.Manager().Queue()

    shr, centroids = sharedArray(centroids)  

    for i in range(NO_OF_THREADS):
        l = i*CHUNK_SIZE
        r = ((i+1)*CHUNK_SIZE) - 1
        pool.apply_async(getSimilarityDistribution, args=(data[l:r], l, shr.name, output_queue,))

    pool.close()
    pool.join()
    shr.unlink()

    similarity_above_90 = 0
    similarity_above_85 = 0
    similarity_above_80 = 0
    similarity_above_75 = 0
    similarity_above_70 = 0
    similarity_above_65 = 0
    similarity_above_60 = 0
    similarity_above_55 = 0
    similarity_above_50 = 0
    similarity_above_40 = 0
    similarity_above_30 = 0
    similarity_above_20 = 0
    similarity_above_10 = 0
    similarity_below_10 = 0
    
    while not output_queue.empty():
        distribution = output_queue.get()
        # print(type(distribution))
        # print(distribution)
        similarity_above_90 += distribution[0]
        similarity_above_85 += distribution[1]
        similarity_above_80 += distribution[2]
        similarity_above_75 += distribution[3]
        similarity_above_70 += distribution[4]
        similarity_above_65 += distribution[5]
        similarity_above_60 += distribution[6]
        similarity_above_55 += distribution[7]
        similarity_above_50 += distribution[8]
        similarity_above_40 += distribution[9]
        similarity_above_30 += distribution[10]
        similarity_above_20 += distribution[11]
        similarity_above_10 += distribution[12]
        similarity_below_10 += distribution[13]
        

    print(similarity_above_90)
    print(similarity_above_85)
    print(similarity_above_80)
    print(similarity_above_75)
    print(similarity_above_70)
    print(similarity_above_65)
    print(similarity_above_60)
    print(similarity_above_55)
    print(similarity_above_50)
    print(similarity_above_40)
    print(similarity_above_30)
    print(similarity_above_20)
    print(similarity_above_10)
    print(similarity_below_10)
    
    # plot the distribution
    plt.plot([90, 85, 80, 75, 70, 65, 60, 55, 50, 40, 30, 20, 10, 10], [similarity_above_90, similarity_above_85, similarity_above_80, similarity_above_75, similarity_above_70, similarity_above_65, similarity_above_60, similarity_above_55, similarity_above_50, similarity_above_40, similarity_above_30, similarity_above_20, similarity_above_10, similarity_below_10])
    plt.xlabel('Similarity (%)')
    plt.ylabel('Count')
    plt.title('Similarity Distribution for ' + label)
    # plt.show()
    plt.savefig("Similarity Distribution for " + label + ".png")

if __name__ == "__main__":
    start_time = time.time()
    
    labels = load_labels(LABLE_FILE_NAME, GLOBAL_N)
    
    label = "research"
    # label2 = "newyork"
    

    data = readData(INPUT_FILE_NAME, labels, label)
    print("Data = " + str(len(data)))
    
    data = preprocessing.normalize(data)
    print("Data Normalized")
    
    dumpToFile(data, NORMALIZED_DATA_FILE_NAME)
    print("Data Dumped")
    
    kmeans = KMeans(n_clusters=200, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    print("KMeans Clustering Done")
    
    dumpToFile(kmeans, KMEANS_CLUSTER_FILE_NAME)
    print("KMeans Cluster Dumped")
    # labels, num_clusters = clusterData(data, 0.7)
    # print("No. of clusters = " + str(num_clusters))

    # writeToBinaryFile(labels, NameError, len(data[0]), data)
    # print("Posting List = " + str(postingList))
    # print("Centroids = " + str(centroids))

    # cluster(data, 2, 0.7)
    printSimilarityDistributionWhileProcessingDataInChunks(data, centroids, label)

    # print(multiprocessing.cpu_count())

    print('The scikit-learn version is {}.'.format(sklearn.__version__))


    elapsed_time = time.time() - start_time
    print("Time Taken = " + str(elapsed_time))