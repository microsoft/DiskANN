import array
import multiprocessing.shared_memory
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
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

INPUT_FILE_NAME = "wikipedia_base.bin"
INPUT_FILE_PICKLE = "wikipedia_base.pickle"
SAMPLE_SIZE_FOR_CLUSTERING = 1000000
NO_OF_CLUSTERS = 200
KMEANS_CLUSTER_FILE_NAME = str(NO_OF_CLUSTERS) + "_kmeans_cluster.pickle"
NORMALIZED_DATA_FILE_NAME = "normalized_data.pickle"
NO_OF_THREADS = 70
GLOBAL_N = 35000000
CHUNK_SIZE = GLOBAL_N//NO_OF_THREADS
COSINE_SIMILARITY_THRESHOLD = 0.75
CENTROIDS_SHAPE = (200, 768)
POSTING_LIST_FILE_NAME = "posting_list_16.pickle"
CENTROID_FILE_NAME = "centroids_16.pickle"
COSINE_SIMILARITY_DEVIATION_THRESHOLD = 0.1 
NUM_OF_ITER_FOR_CLUSTERING = 16

def readData(filename, datatype='f', datatype_size=4):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        # Read the first 8 bytes and unpack them into two 4-byte integers
        N, M = struct.unpack('ii', file.read(8))
        print(N, M)
       
        data = []
        
        # Loop through the number of rows N
        for _ in range(min(N, GLOBAL_N)):
            # Initialize an empty list to store the data
            vector = array.array(datatype)
            # For each row, read M*datatype_size bytes and unpack them into datatype
            vector.fromfile(file, M)
            data.append(vector)

            if(_%1000000==0):
                print(str(_) + " done")
                print("Data Length = " + str(len(data)))

    print("Data Length = " + str(len(data)))
    return data

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

# def assignClusters(l, r):
#     similarity = np_getAngularSimilarity(cosine_similarity(DATA[l:r], CENTROIDS))
#     for i in range(l, r):
#         # Getting Index of bucket which is giving highest similarity
#         max_idx = np.argmax(similarity[i-l])
#         max_val = similarity[i][max_idx]

#         if max_val > COSINE_SIMILARITY_THRESHOLD:
#             LABELS[i] = max_idx
#         else:
#             LOCK.acquire()
#             UNASSIGNED.append(i)
#             LOCK.release()


def assignClusters(data, offset, shr_name, output_queue):
    print("offset = " + str(offset) + " started", flush=True)
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shr_name)
    centroids = np.ndarray(CENTROIDS_SHAPE, dtype=np.float32, buffer=existing_shm.buf)
    # print(centroids.shape)
    # print(CENTROIDS_SHAPE)
    similarity = np_getAngularSimilarity(cosine_similarity(data, centroids))
    postings = [[] for i in range(NO_OF_CLUSTERS)]

    for i in range(len(data)):
        # Getting Index of bucket which is giving highest similarity
        max_idx = np.argmax(similarity[i])
        max_val = similarity[i][max_idx]

        # assignTo = np.where(similarity[i] >= max_val-COSINE_SIMILARITY_DEVIATION_THRESHOLD)

        # for idx in np.nditer(assignTo):
        #     postings[idx].append(i + offset)

        postings[max_idx].append(i + offset)
    
    output_queue.put(postings)
    print("offset = " + str(offset) + " done", flush=True)


def recenterClustersPerCluster(data, output_queue):
    if len(data) == 0:
        output_queue.put(np.zeros(768))
        return
    output_queue.put(data.mean(axis=0))

def recenterClusters(data, postings):
    pool = multiprocessing.Pool(processes=61)
    output_queue = multiprocessing.Manager().Queue()

    for i in range(NO_OF_CLUSTERS):
        pool.apply_async(recenterClustersPerCluster, args=(data[postings[i]], output_queue,))
        print("Cluster = " + str(i) + " started", flush=True)

    pool.close()
    pool.join()

    centroids = []
    while not output_queue.empty():
        centroids.append(output_queue.get())

    return np.array(centroids)

def assignClustersInChunk(data, centroids):
    pool = multiprocessing.Pool(processes=61)
    output_queue = multiprocessing.Manager().Queue()

    shr, centroids = sharedArray(centroids)  

    for i in range(NO_OF_THREADS):
        l = i*CHUNK_SIZE
        r = ((i+1)*CHUNK_SIZE) - 1
        pool.apply_async(assignClusters, args=(data[l:r], l, shr.name, output_queue,))

    pool.close()
    pool.join()
    shr.unlink()

    postings = [[] for i in range(NO_OF_CLUSTERS)]
    while not output_queue.empty():
        cluster = output_queue.get()
        for i in range(NO_OF_CLUSTERS):
            postings[i].extend(cluster[i])

    return postings

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
    similarity_below_50 = 0

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
        else:
            similarity_below_50 += 1

    result = [similarity_above_90, similarity_above_85, similarity_above_80, similarity_above_75, similarity_above_70, similarity_above_65, similarity_above_60, similarity_above_55, similarity_above_50, similarity_below_50]
    # print(result, flush = True)
    output_queue.put(result)
    print("offset = " + str(offset) + " done", flush=True)


def printSimilarityDistributionWhileProcessingDataInChunks(data, centroids):
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
    similarity_below_50 = 0
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
        similarity_below_50 += distribution[9]
        

    print(similarity_above_90)
    print(similarity_above_85)
    print(similarity_above_80)
    print(similarity_above_75)
    print(similarity_above_70)
    print(similarity_above_65)
    print(similarity_above_60)
    print(similarity_above_55)
    print(similarity_above_50)
    print(similarity_below_50)




if __name__ == "__main__":
    start_time = time.time()

    data = loadFromFile(NORMALIZED_DATA_FILE_NAME)
    # centroids = loadFromFile(KMEANS_CLUSTER_FILE_NAME).cluster_centers_
    # posting_list = loadFromFile(POSTING_LIST_FILE_NAME)
    centroids = loadFromFile(CENTROID_FILE_NAME)
    print("Data Loaded")

    printSimilarityDistributionWhileProcessingDataInChunks(data, centroids)

    # executor = ThreadPoolExecutor(max_workers=NO_OF_THREADS)

    # for i in range(NO_OF_THREADS):
    #     l = i*CHUNK_SIZE
    #     r = ((i+1)*CHUNK_SIZE) - 1
    #     executor.submit(assignClusters, l, r)
    
    # executor.shutdown(wait=True)

    # postings = []

    # for i in range(NUM_OF_ITER_FOR_CLUSTERING):
    #     print("Iteration = " + str(i))
    #     postings = assignClustersInChunk(data, centroids)
    #     print("Assigned")

    #     centroids = recenterClusters(data, postings)
    #     print("Recentered")
        
    # dumpToFile(postings, POSTING_LIST_FILE_NAME)
    
    
    # print(postings)


    # print("Unassigned Length = " + str(GLOBAL_N - len(st)))

    # print("Over Assignments = " + str(cnt-GLOBAL_N))

    print(multiprocessing.cpu_count())
   
    print('The scikit-learn version is {}.'.format(sklearn.__version__))


    elapsed_time = time.time() - start_time
    print("Time Taken = " + str(elapsed_time))