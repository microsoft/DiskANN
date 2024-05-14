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


INPUT_FILE_NAME = "wikipedia_base.bin"
SAMPLE_SIZE_FOR_CLUSTERING = 1000000
NO_OF_CLUSTERS = 200
KMEANS_CLUSTER_FILE_NAME = str(NO_OF_CLUSTERS) + "_kmeans_cluster.pickle"

def readData(filename, datatype='f', datatype_size=4):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        # Read the first 8 bytes and unpack them into two 4-byte integers
        N, M = struct.unpack('ii', file.read(8))
        print(N, M)

        randomSampleSpace = set(random.sample(range(N), 1000000))
       
        data = []
        
        # Loop through the number of rows N
        for _ in range(N):
            # Initialize an empty list to store the data
            vector = array.array(datatype)
            # For each row, read M*datatype_size bytes and unpack them into datatype
            vector.fromfile(file, M)
            if _ in randomSampleSpace:
                data.append(vector)

            if(_%1000000==0):
                print(str(_) + " done")
                print("Data Length = " + str(len(data)))

    print("Data Length = " + str(len(data)))
    return data

# Function to covert the similarity to angular similarity
def get_angular_similarity(x):
    x = min(x, 1)
    x = max(x, -1)
    return 1 - math.acos(x) / math.pi


# Vectorized version
np_getAngularSimilarity = np.vectorize(get_angular_similarity)

def writeToBinaryFile(labels, num_clusters, dims, data, centroid_file_name=INPUT_FILE_NAME.strip('.bin')+"_centroids.bin", posting_list_file_name=INPUT_FILE_NAME.strip('.bin')+"_posting_list.bin", datatype='f'):
    posting_list = [[] for i in range(num_clusters)]
    centroids = [np.zeros(dims) for i in range(num_clusters)]
    for i in range(len(labels)):
        posting_list[labels[i]].append(i)
        centroids[labels[i]] = np.add(centroids[labels[i]], data[i])
    
    for i in range(num_clusters):
        centroids[i] = np.divide(centroids[i], len(posting_list[i]))
    
    with open(centroid_file_name, 'wb') as file:
        file.write(struct.pack('ii', num_clusters, dims))
        for i in range(num_clusters):
            file.write(struct.pack(datatype*dims, *centroids[i]))
    
    with open(posting_list_file_name, 'wb') as file:
        for i in range(num_clusters):
            cur_posting_size = len(posting_list[i])
            file.write(struct.pack('i', cur_posting_size))
            file.write(struct.pack('i'*cur_posting_size, *posting_list[i]))

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




if __name__ == "__main__":
    start_time = time.time()

    data = readData(INPUT_FILE_NAME)
    data = preprocessing.normalize(data)
    kmeans = KMeans(n_clusters=200, random_state=0).fit(data)
    dumpToFile(kmeans, KMEANS_CLUSTER_FILE_NAME)
    # labels, num_clusters = clusterData(data, 0.7)
    # print("No. of clusters = " + str(num_clusters))

    # writeToBinaryFile(labels, num_clusters, len(data[0]), data)
    # print("Posting List = " + str(postingList))
    # print("Centroids = " + str(centroids))

    # cluster(data, 2, 0.7)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))


    elapsed_time = time.time() - start_time
    print("Time Taken = " + str(elapsed_time))