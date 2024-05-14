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
NO_OF_CLUSTERS = 200
NO_OF_THREADS = 70
GLOBAL_N = 35000000
CHUNK_SIZE = GLOBAL_N//NO_OF_THREADS
POSTING_LIST_FILE_NAME = "posting_list_16.pickle"
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
    return np.array(data)

def recenterClustersPerCluster(data, output_queue):
    if len(data) == 0:
        output_queue.put(np.zeros(768))
        return
    output_queue.put(data.mean(axis=0))

def recenterClusters(data, postings):
    print(type(postings))

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


def printClusterDistribution(posting_list):
    for i in range(NO_OF_CLUSTERS):
        print(len(posting_list[i]))


def writeToBinaryFile(posting_list, num_clusters, dims, data, centroid_file_name=INPUT_FILE_NAME.strip('.bin')+" "+str(NO_OF_CLUSTERS)+"_centroids.bin", posting_list_file_name=INPUT_FILE_NAME.strip('.bin')+" "+str(NO_OF_CLUSTERS)+"_posting_list.bin", datatype='f'):
    centroids = recenterClusters(data, posting_list)
    
    with open(centroid_file_name, 'wb') as file:
        file.write(struct.pack('ii', num_clusters, dims))
        for i in range(num_clusters):
            file.write(struct.pack(datatype*dims, *centroids[i]))
    
    with open(posting_list_file_name, 'wb') as file:
        for i in range(num_clusters):
            cur_posting_size = len(posting_list[i])
            file.write(struct.pack('i', cur_posting_size))
            file.write(struct.pack('i'*cur_posting_size, *posting_list[i]))

def loadFromFile(filename):
    result = None
    with open(filename, 'rb') as handle:
        result = pickle.load(handle)
    return result

def dumpToFile(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    start_time = time.time()

    # data = readData(INPUT_FILE_NAME)
    # dumpToFile(data, INPUT_FILE_PICKLE)
    posting_list = loadFromFile(POSTING_LIST_FILE_NAME)
    print("Data Loaded")
    # writeToBinaryFile(posting_list, NO_OF_CLUSTERS, len(data[0]), data)
    printClusterDistribution(posting_list)

    elapsed_time = time.time() - start_time
    print("Time Taken = " + str(elapsed_time))