import numpy as np
import matplotlib.pyplot as plt

def read_gt_file(gt_file, npts, lbuild):
    # Read the GT file (assumes distances are stored as float32 after the header and IDs)
    header_size = 8  # 2 integers (npts, lbuild)
    ids_matrix_size = npts * lbuild * 4  # uint32
    distances_offset = header_size + ids_matrix_size

    with open(gt_file, "rb") as f:
        f.seek(distances_offset)  # Skip to the distances
        distances = np.fromfile(f, dtype=np.float32, count=npts * lbuild)
    return distances.reshape(npts, lbuild)

def read_search_result_file(result_file, npts, k):
    # Read the search result file (assumes distances are stored as float32)
    with open(result_file, "rb") as f:
        distances = np.fromfile(f, dtype=np.float32, count=npts * k)
    return distances.reshape(npts, k)

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
    with open(output_file, "w") as f:
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

# Parameters
npts = 5000  # Number of queries
lbuild = 10  # Number of neighbors in GT file
k = 5  # Number of neighbors in search result file
gt_file = "/data/wikipedia/wiki_1m_5k_double_lammetized_gt"
result_file = "/data/wikipedia/result/result_1m_double_page_283_10_dists_float.bin"
percentile_output_file = "/data/wikipedia/percentile_analysis.txt"

# Read distances
gt_distances = read_gt_file(gt_file, npts, lbuild)
search_distances = read_search_result_file(result_file, npts, k)

# Calculate mean differences
mean_differences = calculate_mean_difference(gt_distances, search_distances)
print("Mean differences:", mean_differences)

# Perform percentile analysis and save to file
percentile_analysis(mean_differences, percentile_output_file)

# Plot the mean differences as a dot plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(mean_differences)), mean_differences, label="Mean Distance Differences", color="green", s=10)
plt.xlabel("Query Index")
plt.ylabel("Mean Distance Difference")
plt.title("Mean Distance Differences Between GT and Search Results (Dot Plot)")
plt.legend()
plt.grid()
plt.savefig("/data/wikipedia/mean_distance_differences_dot.png", dpi=300)
plt.show()

print("Done")