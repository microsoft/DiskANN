import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.spatial.distance import cdist

def read_gt_file_dist(gt_file, npts, lbuild):
    # Read the GT file (assumes distances are stored as float32 after the header and IDs)
    header_size = 8  # 2 integers (npts, lbuild)
    ids_matrix_size = npts * lbuild * 4  # uint32
    distances_offset = header_size + ids_matrix_size

    with open(gt_file, "rb") as f:
        f.seek(distances_offset)  # Skip to the distances
        distances = np.fromfile(f, dtype=np.float32, count=npts * lbuild)
    return distances.reshape(npts, lbuild)

def read_gt_file_idx(gt_file, npts, lbuild):
    # Read the GT file (assumes distances are stored as uint32 after the header and IDs)
    header_size = 8  # 2 integers (npts, lbuild)
    
    with open(gt_file, "rb") as f:
        f.seek(header_size)
        ids = np.fromfile(f, dtype=np.uint32, count=npts * lbuild)
    ids = ids.reshape(npts, lbuild)
    return ids

def read_search_result_file(result_file, npts, k):
    # Read the search result file (assumes distances are stored as float32)
    with open(result_file, "rb") as f:
        distances = np.fromfile(f, dtype=np.float32, count=npts * k)
    return distances.reshape(npts, k)

def read_search_result_file_idx(result_file):
    # Read the search result file (assumes distances are stored as uint32 after the header and IDs)
    header_size = 8  # 2 integers (npts, k)
    npts, k = np.fromfile(result_file, dtype=np.uint32, count=2)
    
    with open(result_file, "rb") as f:
        f.seek(header_size)
        ids = np.fromfile(f, dtype=np.uint32, count=npts * k)
    ids = ids.reshape(npts, k)
    return k, ids

def calculate_mean_difference(gt_distances, search_distances, max_valid_distance=1e6):
    # Ensure both arrays have the same number of neighbors (truncate GT to match search result)
    k = search_distances.shape[1]
    gt_distances = gt_distances[:, :k]

    # Replace inf values with NaN to exclude them from calculations
    gt_distances = np.where(gt_distances > max_valid_distance, np.nan, gt_distances)

    # Calculate the absolute difference between GT and search distances
    differences = np.abs(gt_distances - search_distances)

    # Set differences to 0 wherever GT distances were NaN
    differences = np.where(np.isnan(differences), 0, differences)

    # Compute the mean difference for each query
    mean_differences = np.mean(differences, axis=1)
    return mean_differences

def percentile_analysis(mean_differences, output_file):
    # Calculate percentiles from 1 to 100
    percentiles = np.percentile(mean_differences, range(1, 101))
    print("\nPercentile Analysis:")
    
    with open(output_file + 'percentile.txt', "a") as f:
        f.write("Percentile Analysis:\n")
        for i, p in enumerate(percentiles, start=1):
            line = f"Percentile {i}: Mean Distance = {p:.4f}\n"
            print(line.strip())
            f.write(line)

        # Example: Calculate the percentage of queries with mean distance less than a threshold x
        x = 0.5  # Example threshold
        percentage_below_x = np.sum(mean_differences < x) / len(mean_differences) * 100
        threshold_line = f"\nPercentage of queries with mean distance less than {x}: {percentage_below_x:.2f}%\n"
        print(threshold_line.strip())
        f.write(threshold_line)
        
        # plot percentiles
        plt.plot(range(1, 101), percentiles, marker='o')
        plt.xlabel('Percentile')
        plt.ylabel('Mean Distance')
        plt.title('Percentile Analysis of Mean Distances')
        plt.grid()
        plt.savefig(output_file + 'percentile.png', dpi=300)
        plt.close()

    
def calculate_gt_res_distances(gt_indices, search_indices, vectors):
    """
    Calculate distances between corresponding GT and search result points for each query.
    """
    # Ensure GT and search indices have the same shape
    assert gt_indices.shape == search_indices.shape, "GT and search indices must have the same shape."

    # Initialize the distances array
    num_queries, num_neighbors = gt_indices.shape
    distances = np.zeros((num_queries, num_neighbors))

    # Iterate over each query and compute distances
    for i in range(num_queries):
        for j in range(num_neighbors):
            # Get the GT and search vectors
            if (gt_indices[i][j] > len(vectors)):
                #distance is not valid
                distances[i, j] = 0
                continue
            gt_vector = vectors[gt_indices[i][j]]
            search_vector = vectors[search_indices[i][j]]
            
            # print("GT and result same?" , np.array_equal(gt_vector, search_vector))
            
            # if((np.array_equal(gt_vector, search_vector))):
            #     distances[i, j] = 0.0
            #     continue
            # else:
            #     print("GT and result different")

            # Calculate the cosine distance
            numerator = np.dot(gt_vector, search_vector)
            denominator = np.linalg.norm(gt_vector) * np.linalg.norm(search_vector)
            if denominator == 0:
                distances[i, j] = 1.0  # Max cosine distance (completely dissimilar)
            else:
                distances[i, j] = 1 - (numerator / denominator)
            # distances[i, j] = cdist([gt_vector], [search_vector], metric='cosine')[0][0]
    return distances
    
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

# Parameters
npts = 5000  # Number of queries
lbuild = 100  # Number of neighbors in GT file
# k = 5  # Number of neighbors in search result file
gt_file = "/data/wikipedia/gt/wiki_slice1_reduced_double_100_cosine_gt"
# gt_file = '/data/wikipedia/gt/wiki_35m_reduced_double_100_cosine_gt'
# result_file = "/data/wikipedia/result/result_1m_double_page_283_10_dists_float.bin"
# result_file = "/data/wikipedia/result/result_35m_double_313_l2_k5_l10_10_idx_uint32.bin"
result_file = '/data/wikipedia/result/result_slice1_double_014_cosine_k5_l10_10_idx_uint32.bin'
# result_file = '/data/wikipedia/result/result_35m_double_313_l2_k5_l10_10_dists_float.bin'
output_file = "/data/wikipedia/analysis/analysis_014_"
vectors_file = "/data/wikipedia/wiki_vector_1m.bin"

# Read indices
gt_indices = read_gt_file_idx(gt_file, npts, lbuild)
k, search_indices = read_search_result_file_idx(result_file)
# take the first k indices
gt_indices = gt_indices[:, :k]
print("GT Indices shape:", gt_indices.shape)
print("Search Indices shape:", search_indices.shape)
print("GT Indices:", gt_indices[:10])
print("Search Indices :", search_indices[:10])

# get vectors by indices
vectors = readBin(vectors_file, 1000000)
print("Vectors shape:", vectors.shape)
print(vectors[0][:10])

norms = np.linalg.norm(vectors, axis=1)
print("Vector norms (first 10):", norms[:10])
print("Are all vectors normalized?", np.allclose(norms, 1.0))

vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
print("Are all vectors normalized after normalization?", np.allclose(np.linalg.norm(vectors, axis=1), 1.0))

distances = calculate_gt_res_distances(gt_indices, search_indices, vectors)
print("Distances shape:", distances.shape)
print(distances[0][:10])

#save distances
np.save(output_file + 'distances.npy', distances)
# load distances
distances = np.load(output_file + 'distances.npy')
print("Distances shape:", distances.shape)
print(distances[0][:10])

# mean for each query
mean_distances = np.mean(distances, axis=1)
print("Mean Distances shape:", mean_distances.shape)
print(mean_distances[:10])

#max of mean distances
max_mean_distance = np.max(mean_distances)

#min of mean distances
min_mean_distance = np.min(mean_distances)

#std of mean distances
std_mean_distance = np.std(mean_distances)

print("Max Mean Distance:", max_mean_distance)
print("Min Mean Distance:", min_mean_distance)
print("Std Mean Distance:", std_mean_distance)

with open(output_file + 'mean_dist.txt', "w") as f:
    f.write("Mean Distances:\n")
    for i, distance in enumerate(mean_distances):
        line = f"Query {i}: Mean Distance = {distance:.4f}\n"
        # print(line.strip())
        f.write(line)
    
    f.write(f"\nMax Mean Distance: {max_mean_distance:.4f}\n")
    f.write(f"Min Mean Distance: {min_mean_distance:.4f}\n")
    f.write(f"Std Mean Distance: {std_mean_distance:.4f}\n")


# percentile analysis
percentile_analysis(mean_distances, output_file)
# Plot histogram
plt.hist(mean_distances, bins=50, alpha=0.75)
plt.xlabel('Mean Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Distances')
plt.grid()
plt.savefig(output_file + "mean_dist_hist.png", dpi=300)