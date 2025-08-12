#!/usr/bin/env python3
"""
Complete ILP Pipeline Script
This script runs the full pipeline for ILP weight calculation:
1. Query splitting
2. Ground truth calculation (filtered and unfiltered)
3. ILP weight calculation
"""

import argparse
import os
import sys
import subprocess
import shutil

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None

def check_file_exists(filepath, description=""):
    """Check if a file exists and report its status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description} already exists: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"⚠ {description} not found: {filepath}")
        return False

def ensure_executable_exists(executable_path, name=""):
    """Check if an executable exists and is accessible"""
    if not os.path.exists(executable_path):
        print(f"✗ ERROR: {name} executable not found: {executable_path}")
        print("Please ensure DiskANN is built and executables are in the expected location.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run complete ILP pipeline')
    
    # Main argument - workload name
    parser.add_argument('workload', help='Workload name (e.g., TA/NA, PA/EMEA)')
    
    # Optional overrides for file paths
    parser.add_argument('--base_vectors', help='Base vectors file (.u8bin) - overrides default')
    parser.add_argument('--base_labels', help='Base labels file (.txt) - overrides default')
    parser.add_argument('--query_vectors', help='Query vectors file (.u8bin) - overrides default')
    parser.add_argument('--query_labels', help='Query labels file (.txt) - overrides default')
    
    # Base configuration
    parser.add_argument('--base_dir', default='C:\\ads_data', help='Base data directory')
    parser.add_argument('--base_size', default='1M', help='Base dataset size (1M, 40M, etc.)')
    parser.add_argument('--query_size', default='10K', help='Query dataset size (10K, etc.)')
    
    # Parameters
    parser.add_argument('--split_ratio', type=float, default=0.6, help='Train/test split ratio')
    parser.add_argument('--K', type=int, default=100, help='Number of nearest neighbors for ground truth')
    
    # Paths
    parser.add_argument('--output_dir', default=None, help='Output directory (default: same as base files directory + /ilp)')
    parser.add_argument('--diskann_root', default='.', help='DiskANN root directory')
    parser.add_argument('--norm_factors', help='Normalization factors file')
    
    args = parser.parse_args()
    
    # Derive file paths from workload if not explicitly provided
    workload = args.workload
    data_dir = os.path.join(args.base_dir, workload)
    
    # Set default file paths based on workload
    if not args.base_vectors:
        args.base_vectors = os.path.join(data_dir, f"base_vectors_{args.base_size}.u8bin")
    if not args.base_labels:
        args.base_labels = os.path.join(data_dir, f"base_labels_{args.base_size}.txt")
    if not args.query_vectors:
        args.query_vectors = os.path.join(data_dir, f"query_vectors_{args.query_size}.u8bin")
    if not args.query_labels:
        args.query_labels = os.path.join(data_dir, f"query_labels_{args.query_size}.txt")
    
    # Set output directory based on base files location if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(data_dir, f'ilp_{args.query_size}')
    
    # Print configuration
    print(f"ILP Pipeline Configuration:")
    print(f"{'='*50}")
    print(f"Workload: {workload}")
    print(f"Data directory: {data_dir}")
    print(f"Base vectors: {args.base_vectors}")
    print(f"Base labels: {args.base_labels}")
    print(f"Query vectors: {args.query_vectors}")
    print(f"Query labels: {args.query_labels}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*50}")
    
    # Verify files exist
    required_files = [args.base_vectors, args.base_labels, args.query_vectors, args.query_labels]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            print("Please ensure the workload data is available or use explicit file path arguments")
            sys.exit(1)
    
    print("✓ All required files found")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for split data
    query_train_vectors = os.path.join(args.output_dir, "query_vectors.train.uint8bin")
    query_test_vectors = os.path.join(args.output_dir, "query_vectors.test.uint8bin")
    query_train_labels = os.path.join(args.output_dir, "query_labels.train.txt")
    query_test_labels = os.path.join(args.output_dir, "query_labels.test.txt")
    
    # Check for required executables
    compute_gt_exe = os.path.join(args.diskann_root, "x64", "Release", "compute_groundtruth.exe")
    compute_filtered_gt_exe = os.path.join(args.diskann_root, "x64", "Release", "compute_filtered_groundtruth.exe")
    query_split_script = os.path.join(args.diskann_root, "scripts", "ml_ilp", "query_splitting.py")
    norm_factors_script = os.path.join(args.diskann_root, "scripts", "analyze_dataset_span_simple.py")
    ilp_script = os.path.join(args.diskann_root, "scripts", "ml_ilp", "ilp.py")
    
    if not ensure_executable_exists(compute_gt_exe, "compute_groundtruth"):
        sys.exit(1)
    if not ensure_executable_exists(compute_filtered_gt_exe, "compute_filtered_groundtruth"):
        sys.exit(1)
    if not ensure_executable_exists(query_split_script, "query_splitting.py"):
        sys.exit(1)
    
    # Step 1: Split queries into train/test (skip if already done)
    print("STEP 1: Splitting queries into train/test sets")
    
    if (check_file_exists(query_train_vectors, "Training query vectors") and 
        check_file_exists(query_test_vectors, "Test query vectors") and
        check_file_exists(query_train_labels, "Training query labels") and
        check_file_exists(query_test_labels, "Test query labels")):
        print("✓ Query splitting already completed, skipping...")
    else:
        split_cmd = [
            sys.executable, 
            query_split_script,
            "--input_bin", args.query_vectors,
            "--input_label", args.query_labels,
            "--output_bin_prefix", os.path.join(args.output_dir, "query_vectors"),
            "--output_label_prefix", os.path.join(args.output_dir, "query_labels"),
            "--split_ratio", str(args.split_ratio),
            "--data_type", "uint8"
        ]
        
        result = run_command(split_cmd, "Query splitting")
        if result is None:
            print("Query splitting failed. Exiting.")
            sys.exit(1)
    
    # Step 2: Calculate unfiltered ground truth using compute_groundtruth
    print("STEP 2: Calculating unfiltered ground truth")
    
    unfiltered_gt_path = os.path.join(args.output_dir, f"unfiltered_groundtruth_{args.base_size}_{args.query_size}_train.bin")
    unfiltered_match_scores_path = os.path.join(args.output_dir, f"unfiltered_match_scores_{args.base_size}_{args.query_size}_test.txt")
    
    if (check_file_exists(unfiltered_gt_path, "Unfiltered ground truth") and 
        check_file_exists(unfiltered_match_scores_path, "Unfiltered match scores")):
        print("✓ Unfiltered ground truth already computed, skipping...")
    else:
        # Command for unfiltered ground truth
        search_unfiltered_cmd = [
            compute_gt_exe,
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--base_file", args.base_vectors,
            "--query_file", query_train_vectors,
            "--base_label_file", args.base_labels,
            "--query_label_file", query_train_labels,
            "--gt_file", unfiltered_gt_path,
            "--K", str(args.K),
            "--match_score_file", unfiltered_match_scores_path
        ]
        
        result = run_command(search_unfiltered_cmd, "Unfiltered ground truth calculation")
        if result is None:
            print("Unfiltered ground truth calculation failed. Exiting.")
            sys.exit(1)
    
    # Step 3: Calculate filtered ground truth
    print("STEP 3: Calculating filtered ground truth")
    
    filtered_gt_path = os.path.join(args.output_dir, f"filtered_groundtruth_{args.base_size}_{args.query_size}_train.bin")
    
    if check_file_exists(filtered_gt_path, "Filtered ground truth"):
        print("✓ Filtered ground truth already computed, skipping...")
    else:
        # Command for filtered ground truth - using correct executable and parameters
        search_filtered_cmd = [
            compute_filtered_gt_exe,
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--base_file", args.base_vectors,
            "--query_file", query_train_vectors,
            "--gt_file", filtered_gt_path,
            "--base_labels", args.base_labels,
            "--query_labels", query_train_labels,
            "--K", str(args.K)
        ]
        
        result = run_command(search_filtered_cmd, "Filtered ground truth calculation")
        if result is None:
            print("Filtered ground truth calculation failed. Exiting.")
            sys.exit(1)
    
    # Step 3.5: Generate normalization factors for L2 distance if not provided
    if not args.norm_factors:
        print("STEP 3.5: Generating normalization factors for L2 distance")
        
        norm_factors_path = os.path.join(data_dir, f"normalization_factors_{args.base_size}.txt")
        
        if check_file_exists(norm_factors_path, "Normalization factors"):
            args.norm_factors = norm_factors_path
            print("✓ Using existing normalization factors")
        elif ensure_executable_exists(norm_factors_script, "analyze_dataset_span_simple.py"):
            norm_cmd = [
                sys.executable,
                norm_factors_script,
                args.base_vectors,
                "--metric", "euclidean",
                "--samples", "10000",
                "--dtype", "uint8",
                "--norm-factors", norm_factors_path,
                "--seed", "42"
            ]
            
            result = run_command(norm_cmd, "Normalization factors generation")
            if result is not None:
                args.norm_factors = norm_factors_path
                print(f"✓ Generated normalization factors: {norm_factors_path}")
            else:
                print("⚠ Normalization factors generation failed, continuing without them")
        else:
            print("⚠ Normalization script not found, continuing without normalization factors")
    
    # Step 4: Run ILP weight calculation
    print("STEP 4: Running ILP weight calculation")
    
    if not ensure_executable_exists(ilp_script, "ilp.py"):
        print("⚠ ILP script not found, skipping weight calculation")
        print("You can run ILP manually later using the generated ground truth files")
    else:
        ilp_cmd = [
            sys.executable,
            ilp_script,
            unfiltered_gt_path,
            filtered_gt_path,
            # Note: match scores file not available, ILP script will work without it
            "--method", "pulp",  # Start with simple pulp method
            "--eps", "0.001"
        ]
        
        if args.norm_factors:
            ilp_cmd.extend(["--norm_factors", args.norm_factors])
        
        result = run_command(ilp_cmd, "ILP weight calculation")
        if result is None:
            print("ILP weight calculation failed. Trying with different parameters...")
            
            # Try with PuLP method if ratio fails
            ilp_cmd_pulp = ilp_cmd.copy()
            try:
                ilp_cmd_pulp[ilp_cmd_pulp.index("ratio")] = "pulp"
                result = run_command(ilp_cmd_pulp, "ILP weight calculation (PuLP method)")
            except ValueError:
                # "ratio" not in list, already using pulp
                print("Already using PuLP method, trying with different epsilon...")
                ilp_cmd_eps = ilp_cmd.copy()
                ilp_cmd_eps[ilp_cmd_eps.index("0.001")] = "0.01"
                result = run_command(ilp_cmd_eps, "ILP weight calculation (larger epsilon)")
        
        if result is None:
            print("⚠ All ILP methods failed. Check your data and try manually.")
            print("Ground truth files are ready for manual ILP execution:")
            print(f"  Unfiltered GT: {unfiltered_gt_path}")
            print(f"  Filtered GT: {filtered_gt_path}")
        else:
            print("✓ ILP weight calculation completed successfully!")
    
    # Save results summary
    summary_path = os.path.join(args.output_dir, "pipeline_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ILP Pipeline Results Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Input Data:\n")
        f.write(f"  Base vectors: {args.base_vectors}\n")
        f.write(f"  Base labels: {args.base_labels}\n")
        f.write(f"  Query vectors: {args.query_vectors}\n")
        f.write(f"  Query labels: {args.query_labels}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Split ratio: {args.split_ratio}\n")
        f.write(f"  K (neighbors): {args.K}\n\n")
        f.write(f"Generated Files:\n")
        f.write(f"  Train queries: {query_train_vectors} ({'exists' if os.path.exists(query_train_vectors) else 'missing'})\n")
        f.write(f"  Test queries: {query_test_vectors} ({'exists' if os.path.exists(query_test_vectors) else 'missing'})\n")
        f.write(f"  Unfiltered GT: {unfiltered_gt_path} ({'exists' if os.path.exists(unfiltered_gt_path) else 'missing'})\n")
        f.write(f"  Filtered GT: {filtered_gt_path} ({'exists' if os.path.exists(filtered_gt_path) else 'missing'})\n")
        if args.norm_factors:
            f.write(f"  Normalization factors: {args.norm_factors} ({'exists' if os.path.exists(args.norm_factors) else 'missing'})\n")
        f.write(f"\n")
        if 'result' in locals() and result:
            f.write("ILP Results:\n")
            f.write(result.stdout)
        f.write(f"\nPipeline completed at: {__import__('datetime').datetime.now()}\n")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in: {args.output_dir}")
    print(f"Summary saved in: {summary_path}")
    
    # Print status of key files
    print(f"\nGenerated Files Status:")
    print(f"  Query splits: {'✓' if os.path.exists(query_test_vectors) else '✗'}")
    print(f"  Unfiltered GT: {'✓' if os.path.exists(unfiltered_gt_path) else '✗'}")
    print(f"  Filtered GT: {'✓' if os.path.exists(filtered_gt_path) else '✗'}")
    if args.norm_factors:
        print(f"  Norm factors: {'✓' if os.path.exists(args.norm_factors) else '✗'}")
    
    if os.path.exists(unfiltered_gt_path) and os.path.exists(filtered_gt_path):
        print(f"\n✓ Ready for ILP weight calculation!")
        print(f"Check the summary file for ILP weights (w_d, w_m)")
    else:
        print(f"\n⚠ Some ground truth files are missing. Check the log above for errors.")

if __name__ == "__main__":
    main()
