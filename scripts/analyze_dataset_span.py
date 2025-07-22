#!/usr/bin/env python3
"""
Script to analyze the span of a dataset in terms of distance.
Samples points and calculates distance statistics for normalization.
"""

import argparse
import struct
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import random


def read_binary_file(file_path, dtype=np.float32):
    """
    Read a binary file in DiskANN format.
    Returns: (data, num_points, dimensions)
    """
    with open(file_path, 'rb') as f:
        # Read header: npts (int32), ndims (int32)
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("File too short to contain header")
        npts, ndims = struct.unpack('ii', header)
        
        print(f"Reading {npts} points with {ndims} dimensions each.")
        
        # Read the data
        data = np.fromfile(f, dtype=dtype, count=npts * ndims).reshape(npts, ndims)
    
    return data, npts, ndims


def sample_distances(data, num_samples=10000, metric='euclidean', seed=42):
    """
    Sample random pairs of points and compute distances.
    
    Args:
        data: numpy array of shape (n_points, n_dims)
        num_samples: number of distance samples to compute
        metric: 'euclidean', 'cosine', or 'inner_product'
        seed: random seed for reproducibility
    
    Returns:
        distances: array of sampled distances
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_points = data.shape[0]
    distances = []
    
    print(f"Sampling {num_samples} distance pairs using {metric} metric...")
    
    for _ in tqdm(range(num_samples)):
        # Sample two random points
        i, j = np.random.choice(n_points, 2, replace=False)
        
        if metric == 'euclidean':
            dist = np.linalg.norm(data[i] - data[j])
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
            dist = 1 - cos_sim
        elif metric == 'inner_product':
            # For inner product, we use negative dot product as distance
            dist = -np.dot(data[i], data[j])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        distances.append(dist)
    
    return np.array(distances)


def analyze_distance_span(distances):
    """
    Analyze the distribution of distances and return statistics.
    """
    stats = {
        'min': np.min(distances),
        'max': np.max(distances),
        'mean': np.mean(distances),
        'std': np.std(distances),
        'median': np.median(distances),
        'q25': np.percentile(distances, 25),
        'q75': np.percentile(distances, 75),
        'q95': np.percentile(distances, 95),
        'q99': np.percentile(distances, 99),
        'span': np.max(distances) - np.min(distances)
    }
    
    return stats


def suggest_normalization_factors(stats):
    """
    Suggest normalization factors based on distance statistics.
    """
    suggestions = {}
    
    # Method 1: Normalize by span (max - min)
    suggestions['span_norm'] = 1.0 / stats['span']
    
    # Method 2: Normalize by standard deviation
    suggestions['std_norm'] = 1.0 / stats['std']
    
    # Method 3: Normalize by 95th percentile (robust to outliers)
    suggestions['q95_norm'] = 1.0 / stats['q95']
    
    # Method 4: Normalize by mean
    suggestions['mean_norm'] = 1.0 / stats['mean']
    
    # Method 5: Z-score normalization factor (mean centering)
    suggestions['zscore_shift'] = -stats['mean']
    suggestions['zscore_scale'] = 1.0 / stats['std']
    
    return suggestions


def plot_distance_distribution(distances, output_path=None):
    """
    Plot the distribution of distances.
    """
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(distances, bins=50, alpha=0.7, density=True)
    plt.title('Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(distances)
    plt.title('Distance Box Plot')
    plt.ylabel('Distance')
    
    # Q-Q plot against normal distribution
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(distances, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    
    # CDF
    plt.subplot(2, 2, 4)
    sorted_distances = np.sort(distances)
    p = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    plt.plot(sorted_distances, p)
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Distance')
    plt.ylabel('Cumulative Probability')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset span in terms of distance')
    parser.add_argument('data_file', help='Binary data file path')
    parser.add_argument('--metric', choices=['euclidean', 'cosine', 'inner_product'], 
                       default='euclidean', help='Distance metric to use')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of distance samples to compute')
    parser.add_argument('--dtype', choices=['float32', 'uint8', 'int8'], default='float32',
                       help='Data type of the binary file')
    parser.add_argument('--plot', action='store_true', help='Generate distance distribution plots')
    parser.add_argument('--plot-output', help='Output path for plots (if not specified, shows interactive plot)')
    parser.add_argument('--output', help='Output file to save statistics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set numpy data type
    dtype_map = {
        'float32': np.float32,
        'uint8': np.uint8,
        'int8': np.int8
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Analyzing dataset: {args.data_file}")
    print(f"Distance metric: {args.metric}")
    print(f"Number of samples: {args.samples}")
    print(f"Data type: {args.dtype}")
    print("-" * 50)
    
    # Read the data
    try:
        data, npts, ndims = read_binary_file(args.data_file, dtype)
        print(f"Successfully loaded {npts} points with {ndims} dimensions")
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)
    
    # Convert to float for distance calculations if needed
    if dtype != np.float32:
        data = data.astype(np.float32)
    
    # Sample distances
    distances = sample_distances(data, args.samples, args.metric, args.seed)
    
    # Analyze distance distribution
    stats = analyze_distance_span(distances)
    
    # Get normalization suggestions
    suggestions = suggest_normalization_factors(stats)
    
    # Print results
    print("\n" + "="*60)
    print("DISTANCE STATISTICS")
    print("="*60)
    
    for key, value in stats.items():
        print(f"{key.upper():>12}: {value:.6f}")
    
    print("\n" + "="*60)
    print("NORMALIZATION SUGGESTIONS")
    print("="*60)
    
    print(f"{'Method':<20} {'Factor':<15} {'Description'}")
    print("-" * 60)
    print(f"{'Span normalization':<20} {suggestions['span_norm']:<15.6f} {'1 / (max - min)'}")
    print(f"{'Std normalization':<20} {suggestions['std_norm']:<15.6f} {'1 / std'}")
    print(f"{'95th percentile':<20} {suggestions['q95_norm']:<15.6f} {'1 / 95th percentile'}")
    print(f"{'Mean normalization':<20} {suggestions['mean_norm']:<15.6f} {'1 / mean'}")
    print(f"{'Z-score (shift)':<20} {suggestions['zscore_shift']:<15.6f} {'-mean (for centering)'}")
    print(f"{'Z-score (scale)':<20} {suggestions['zscore_scale']:<15.6f} {'1 / std (for scaling)'}")
    
    # Recommended w_m values for different approaches
    print("\n" + "="*60)
    print("RECOMMENDED w_m VALUES FOR FILTER WEIGHTING")
    print("="*60)
    
    # Assume filter similarity is in range [0, 1], we want it to have similar scale to distances
    print(f"{'Approach':<25} {'w_m suggestion':<15} {'Rationale'}")
    print("-" * 65)
    print(f"{'Conservative (10%)':<25} {stats['mean'] * 0.1:<15.6f} {'10% of mean distance'}")
    print(f"{'Moderate (50%)':<25} {stats['mean'] * 0.5:<15.6f} {'50% of mean distance'}")
    print(f"{'Aggressive (100%)':<25} {stats['mean']:<15.6f} {'Equal to mean distance'}")
    print(f"{'Very aggressive (200%)':<25} {stats['mean'] * 2.0:<15.6f} {'2x mean distance'}")
    
    # Save results if requested
    if args.output:
        results = {
            'statistics': stats,
            'normalization_factors': suggestions,
            'metadata': {
                'data_file': args.data_file,
                'metric': args.metric,
                'num_samples': args.samples,
                'num_points': npts,
                'dimensions': ndims,
                'dtype': args.dtype
            }
        }
        
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Generate plots if requested
    if args.plot:
        try:
            plot_distance_distribution(distances, args.plot_output)
        except ImportError as e:
            print(f"Warning: Could not generate plots. Missing dependencies: {e}")
        except Exception as e:
            print(f"Error generating plots: {e}")


if __name__ == '__main__':
    main()
