#!/usr/bin/env python3
"""
Simple script to analyze the span of a dataset in terms of distance.
No external dependencies except numpy.
"""

import argparse
import struct
import numpy as np
import sys
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
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_points = data.shape[0]
    distances = []
    
    print(f"Sampling {num_samples} distance pairs using {metric} metric...")
    
    # Simple progress indicator
    progress_step = max(1, num_samples // 20)
    
    for i in range(num_samples):
        if i % progress_step == 0:
            print(f"Progress: {i}/{num_samples} ({100*i/num_samples:.1f}%)")
        
        # Sample two random points
        idx1, idx2 = np.random.choice(n_points, 2, replace=False)
        
        if metric == 'euclidean':
            dist = np.linalg.norm(data[idx1] - data[idx2])
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = np.dot(data[idx1], data[idx2]) / (np.linalg.norm(data[idx1]) * np.linalg.norm(data[idx2]))
            dist = 1 - cos_sim
        elif metric == 'inner_product':
            # For inner product, we use negative dot product as distance
            dist = -np.dot(data[idx1], data[idx2])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        distances.append(dist)
    
    print("Sampling complete!")
    return np.array(distances)


def analyze_distance_span(distances):
    """
    Analyze the distribution of distances and return statistics.
    """
    stats = {
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'median': float(np.median(distances)),
        'q25': float(np.percentile(distances, 25)),
        'q75': float(np.percentile(distances, 75)),
        'q95': float(np.percentile(distances, 95)),
        'q99': float(np.percentile(distances, 99)),
        'span': float(np.max(distances) - np.min(distances))
    }
    
    return stats


def suggest_normalization_factors(stats):
    """
    Suggest normalization factors to bring distances to [0,1] scale,
    so they match the magnitude of filter similarity scores.
    """
    suggestions = {}
    
    # Method 1: Normalize by span (max - min) to get [0,1] range
    suggestions['span_norm'] = 1.0 / stats['span']
    suggestions['span_shift'] = -stats['min']  # Shift to start from 0
    
    # Method 2: Normalize by standard deviation (assumes roughly normal distribution)
    suggestions['std_norm'] = 1.0 / stats['std']
    suggestions['std_shift'] = -stats['mean']  # Center around 0
    
    # Method 3: Robust normalization using percentiles (handles outliers better)
    suggestions['robust_norm'] = 1.0 / (stats['q95'] - stats['q25'])
    suggestions['robust_shift'] = -stats['q25']
    
    # Method 4: Simple mean normalization (most common distances become ~1)
    suggestions['mean_norm'] = 1.0 / stats['mean']
    
    # Method 5: Median normalization (robust to outliers)
    suggestions['median_norm'] = 1.0 / stats['median']
    
    return suggestions


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset span in terms of distance')
    parser.add_argument('data_file', help='Binary data file path')
    parser.add_argument('--metric', choices=['euclidean', 'cosine', 'inner_product'], 
                       default='euclidean', help='Distance metric to use')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of distance samples to compute')
    parser.add_argument('--dtype', choices=['float32', 'uint8', 'int8'], default='float32',
                       help='Data type of the binary file')
    parser.add_argument('--output', help='Output JSON file to save detailed statistics')
    parser.add_argument('--norm-factors', help='Output text file for normalization factors (default: normalization_factors.txt)')
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
    print("NORMALIZATION FACTORS")
    print("="*60)
    print("Goal: Transform distances to [0,1] scale to match filter similarity magnitude")
    
    print(f"\n{'Method':<25} {'Scale Factor':<15} {'Shift Factor':<15} {'Usage'}")
    print("-" * 80)
    print(f"{'Min-Max normalization':<25} {suggestions['span_norm']:<15.6f} {suggestions['span_shift']:<15.6f} {'(dist + shift) * scale'}")
    print(f"{'Robust (IQR) norm':<25} {suggestions['robust_norm']:<15.6f} {suggestions['robust_shift']:<15.6f} {'Handles outliers well'}")
    print(f"{'Standard deviation':<25} {suggestions['std_norm']:<15.6f} {suggestions['std_shift']:<15.6f} {'Assumes normal distribution'}")
    print(f"{'Mean normalization':<25} {suggestions['mean_norm']:<15.6f} {'0.0':<15} {'Simple scaling only'}")
    print(f"{'Median normalization':<25} {suggestions['median_norm']:<15.6f} {'0.0':<15} {'Robust to outliers'}")
    
    
    # Always save simple normalization factors to text file
    norm_config_file = args.norm_factors if args.norm_factors else 'normalization_factors.txt'
    
    # Save simple text format: scale_factor shift_factor
    with open(norm_config_file, 'w') as f:
        f.write(f"{suggestions['span_norm']:.10f} {suggestions['span_shift']:.10f}\n")
    
    print(f"\nNormalization factors saved to: {norm_config_file}")
    print(f"Content: scale={suggestions['span_norm']:.10f}, shift={suggestions['span_shift']:.10f}")
    print(f"Usage: normalized_distance = (distance + {suggestions['span_shift']:.6f}) * {suggestions['span_norm']:.6f}")
    print(f"Use this file with --normalization_factors parameter in build/search commands.")
    
    # Save detailed results if requested
    if args.output:
        results = {
            'statistics': stats,
            'normalization_factors': suggestions,
            'recommended_approach': {
                'method': 'min_max',
                'scale_factor': suggestions['span_norm'],
                'shift_factor': suggestions['span_shift'],
                'formula': '(distance - min) * scale_factor',
                'description': 'Maps distances to [0,1] range to match filter similarity scale'
            },
            'alternative_approaches': {
                'robust': {
                    'scale_factor': suggestions['robust_norm'],
                    'shift_factor': suggestions['robust_shift'],
                    'description': 'Uses IQR for better outlier handling'
                },
                'mean': {
                    'scale_factor': suggestions['mean_norm'],
                    'shift_factor': 0.0,
                    'description': 'Simple scaling, average distance becomes 1.0'
                }
            },
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
        print(f"Detailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
