#!/usr/bin/env python3
"""
Utility to read and apply normalization factors from config file.
"""

import json
import sys
import os


class NormalizationConfig:
    """Class to handle normalization configuration."""
    
    def __init__(self, config_file='normalization_config.json'):
        """Load normalization config from file."""
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Normalization config file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_file}: {e}")
    
    def get_recommended_params(self):
        """Get recommended normalization parameters."""
        return self.config['recommended']
    
    def get_alternative_params(self, method='robust'):
        """Get alternative normalization parameters."""
        if method not in self.config['alternatives']:
            available = list(self.config['alternatives'].keys())
            raise ValueError(f"Method '{method}' not available. Available: {available}")
        return self.config['alternatives'][method]
    
    def normalize_distance(self, distance, method='recommended'):
        """Apply normalization to a distance value."""
        if method == 'recommended':
            params = self.get_recommended_params()
        else:
            params = self.get_alternative_params(method)
        
        # Apply: (distance + shift_factor) * scale_factor
        normalized = (distance + params['shift_factor']) * params['scale_factor']
        return normalized
    
    def normalize_distances(self, distances, method='recommended'):
        """Apply normalization to an array of distance values."""
        if hasattr(distances, '__iter__'):
            return [self.normalize_distance(d, method) for d in distances]
        else:
            return self.normalize_distance(distances, method)
    
    def get_distance_stats(self):
        """Get original distance statistics."""
        return self.config['distance_stats']
    
    def get_metadata(self):
        """Get metadata about the config generation."""
        return self.config['metadata']
    
    def print_summary(self):
        """Print a summary of the normalization config."""
        print("Normalization Configuration Summary")
        print("=" * 50)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Metric: {self.config['metric']}")
        
        stats = self.get_distance_stats()
        print(f"\nDistance Statistics:")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        
        rec = self.get_recommended_params()
        print(f"\nRecommended Normalization:")
        print(f"  Method: {rec['method']}")
        print(f"  Scale Factor: {rec['scale_factor']:.6f}")
        print(f"  Shift Factor: {rec['shift_factor']:.6f}")
        print(f"  Formula: {rec['formula']}")
        
        print(f"\nAlternative Methods Available:")
        for method in self.config['alternatives']:
            params = self.config['alternatives'][method]
            print(f"  {method}: scale={params['scale_factor']:.6f}, shift={params['shift_factor']:.6f}")


def main():
    """Command line interface for normalization utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalization utility')
    parser.add_argument('--config', default='normalization_config.json',
                       help='Normalization config file path')
    parser.add_argument('--summary', action='store_true',
                       help='Print config summary')
    parser.add_argument('--test-distance', type=float,
                       help='Test normalization on a distance value')
    parser.add_argument('--method', default='recommended',
                       choices=['recommended', 'robust', 'mean', 'median'],
                       help='Normalization method to use')
    
    args = parser.parse_args()
    
    try:
        norm_config = NormalizationConfig(args.config)
        
        if args.summary:
            norm_config.print_summary()
        
        if args.test_distance is not None:
            normalized = norm_config.normalize_distance(args.test_distance, args.method)
            print(f"Original distance: {args.test_distance}")
            print(f"Normalized distance ({args.method}): {normalized:.6f}")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
