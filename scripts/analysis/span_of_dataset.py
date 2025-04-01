import numpy as np
import struct
import os
from scipy.spatial.distance import cosine
from tqdm import tqdm
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count

#calculate the distance between all the vector pairs form the wiki_vector_1m.bin file and give us the max distance

def readBin(bin_file, n):
    # Read the binary file and return the vectors
    with open(bin_file, 'rb') as f:
        # Read the header (first 8 bytes)
        header = f.read(8)
        num_vectors, vector_size = struct.unpack('ii', header)
        num_vectors = min(num_vectors, n)  # Limit to n vectors
        
        print(f"Header values - num_vectors: {num_vectors}, vector_size: {vector_size}")
        
        # Read the remaining n data (excluding the header)
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * vector_size)
        
        # # Check if the data size matches the expected size
        # expected_size = num_vectors * vector_size * 4  # 4 bytes per float32
        # actual_size = f.tell() - 8  # Subtract header size
        # print(f"Expected size: {expected_size}, Actual size: {actual_size}")
        
        # if actual_size != expected_size:
        #     raise ValueError(f"Data size mismatch! Expected {expected_size}, but got {actual_size}.")
        
        # Reshape the data into vectors
        vectors = data.reshape(num_vectors, vector_size)
    
    return vectors

# def calculate_distance(vectors, distance_file='/data/wikipedia/analysis/vector_wise_distance_slice.txt', batch_size=1000):
#     vectors = np.array(vectors)  # Ensure it's a NumPy array
#     num_vectors = len(vectors)
    
#     # Initialize storage for max distances
#     max_distances = []

#     with open(distance_file, 'w') as f:
#         f.write(f"Number of vectors: {num_vectors}\n")
#         f.write(f"Vector size: {vectors.shape[1]}\n")

#         # Compute distances in batches
#         for i in tqdm(range(0, num_vectors, batch_size), desc="Processing batches", dynamic_ncols=True):
#             batch_vectors = vectors[i:i + batch_size]
            
#             # Compute cosine distances for the current batch
#             distances_cosine = cdist(batch_vectors, vectors, metric='cosine')
            
#             # Save the max distance for each vector in the batch
#             max_distances.extend(np.max(distances_cosine, axis=1))
            
#             # Optionally, write distances to the file (commented out for large datasets)
#             # for row in distances_cosine:
#             #     f.write(" ".join(map(str, row)) + "\n")
    
#     return np.array(max_distances)

def process_batch(args):
    """
    Compute distances for a single batch.
    """
    batch_start, vectors, batch_size = args
    batch_vectors = vectors[batch_start:batch_start + batch_size]
    distances_cosine = cdist(batch_vectors, vectors, metric='cosine')
    return np.max(distances_cosine, axis=1)

def calculate_distance(vectors, distance_file='/data/wikipedia/analysis/vector_wise_distance_35m_014.txt', batch_size=100):
    vectors = np.array(vectors)  # Ensure it's a NumPy array
    num_vectors = len(vectors)
    
    # Initialize storage for max distances
    max_distances = []
    print("CPU count:", cpu_count())
    # Create a pool of workers
    with Pool(processes=cpu_count()) as pool:
        # Divide the work into batches
        batch_starts = range(0, num_vectors, batch_size)
        
        # Prepare arguments for each batch
        args = [(batch_start, vectors, batch_size) for batch_start in batch_starts]
        
        # Process batches in parallel
        results = list(tqdm(pool.imap(process_batch, args), 
                            total=len(batch_starts), desc="Processing batches", dynamic_ncols=True))
        
        # Flatten the results
        for result in results:
            max_distances.extend(result)

    # Save the results to a file
    with open(distance_file, 'w') as f:
        f.write(f"Number of vectors: {num_vectors}\n")
        f.write(f"Vector size: {vectors.shape[1]}\n")
        f.write(f"Max distances (Cosine): {max_distances}\n")
    
    return np.array(max_distances)

def main():
    bin_file = "/data/wikipedia/wiki_vector.bin"
    print("Reading vectors from binary file...")
    vectors = readBin(bin_file, 1000000)  # Read up to 1 million vectors
    print("Read vectors from binary file")
    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector size: {len(vectors[0])}")
    
    print("Calculating distances...")
    distances_cosine = calculate_distance(vectors)
    print("Distances calculated")
    
    # Get the maximum distance for each vector
    max_distance_cosine = np.max(distances_cosine)
    
    # Calculate additional metrics
    max_of_max = np.max(distances_cosine)
    mean_of_max = np.mean(distances_cosine)
    min_of_max = np.min(distances_cosine)
    std_of_max = np.std(distances_cosine)
    variance_of_max = np.var(distances_cosine)
    range_of_max = max_of_max - min_of_max
    percentiles = np.percentile(distances_cosine, [1, 25, 50, 75, 99])  # 1st, 25th, 50th (median), 75th, 99th percentiles

    # Save the metrics to a file
    with open("/data/wikipedia/analysis/vector_wise_max_distance_35m_014.txt", "w") as f:
        f.write(f"Max of max distances (Cosine): {max_of_max}\n")
        f.write(f"Mean of max distances (Cosine): {mean_of_max}\n")
        f.write(f"Min of max distances (Cosine): {min_of_max}\n")
        f.write(f"Std of max distances (Cosine): {std_of_max}\n")
        f.write(f"Variance of max distances (Cosine): {variance_of_max}\n")
        f.write(f"Range of max distances (Cosine): {range_of_max}\n")
        f.write(f"1st Percentile (Cosine): {percentiles[0]}\n")
        f.write(f"25th Percentile (Cosine): {percentiles[1]}\n")
        f.write(f"Median (50th Percentile, Cosine): {percentiles[2]}\n")
        f.write(f"75th Percentile (Cosine): {percentiles[3]}\n")
        f.write(f"99th Percentile (Cosine): {percentiles[4]}\n")
    
    # Print the metrics to the console
    print("Metrics:")
    print(f"Max of max distances (Cosine): {max_of_max}")
    print(f"Mean of max distances (Cosine): {mean_of_max}")
    print(f"Min of max distances (Cosine): {min_of_max}")
    print(f"Std of max distances (Cosine): {std_of_max}")
    print(f"Variance of max distances (Cosine): {variance_of_max}")
    print(f"Range of max distances (Cosine): {range_of_max}")
    print(f"1st Percentile (Cosine): {percentiles[0]}")
    print(f"25th Percentile (Cosine): {percentiles[1]}")
    print(f"Median (50th Percentile, Cosine): {percentiles[2]}")
    print(f"75th Percentile (Cosine): {percentiles[3]}")
    print(f"99th Percentile (Cosine): {percentiles[4]}")
    
    print("Max distances and metrics saved to file") 
    
if __name__ == "__main__":
    main()      
                