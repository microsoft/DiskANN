import argparse
import struct
import numpy as np
import sys
from tqdm import tqdm
from gekko import GEKKO
import pulp


def read_ground_truth(file_path):
    """
    Reads a ground truth file saved in binary format.
    The file contains:
    - npts (int32): number of queries
    - ndims (int32): number of nearest neighbors per query (K)
    - npts * ndims int32: ground truth indices
    - npts * ndims float32: distances
    Returns:
    - indices: numpy array of shape (npts, ndims) containing ground truth indices
    - distances: numpy array of shape (npts, ndims) containing distances
    """
    with open(file_path, 'rb') as f:
        # Read the header
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("File too short to contain header")
        npts, ndims = struct.unpack('ii', header)
        
        print(f"Reading {npts} queries with {ndims} nearest neighbors each.")

        # Read the ground truth indices
        indices = np.fromfile(f, dtype=np.int32, count=npts * ndims).reshape(npts, ndims)

        # Read the distances
        distances = np.fromfile(f, dtype=np.float32, count=npts * ndims).reshape(npts, ndims)

    return indices, distances

def direct_ratio_method(distances, matches, eps=1e-4):
    Q, N = distances.shape
    max_diff = 0.0
    total_pairs = 0
    for q in range(Q):
        d = distances[q]
        m = matches[q]
        pos_idx = np.where(m == 1)[1]
        neg_idx = np.where(m == 0)[0]
        for i in pos_idx:
            for j in neg_idx:
                total_pairs += 1
                diff = d[i] - d[j] + eps
                if diff > max_diff:
                    max_diff = diff
    w_d, w_m = 1.0, max_diff
    return w_d, w_m, total_pairs, 0


def lp_soft_method_gekko(distances, matches, eps=1e-4):
    Q, N = distances.shape
    # Build LP
    # Using GEKKO
    print("using GEKKO")
    m = GEKKO(remote=False)
    w_d = 1
    w_m = m.Var(lb=0, name='w_m')
    slacks = []
    for q in tqdm(range(Q), desc="Building LP constraints"):
        d = distances[q]
        mvals = matches[q]
        pos = np.where(mvals == 1)[0]
        neg = np.where(mvals == 0)[0]
        for i in pos:
            neg_sample = np.random.choice(neg, size=min(1, len(neg)), replace=False)
            for j in neg_sample:
                s = m.Var(lb=0)
                slacks.append(s)
                m.Equation(w_d*d[i] + w_m*(1-mvals[i]) + eps <= w_d*d[j] + w_m*(1-mvals[j]) + s)

    print(f"Total equations: {len(slacks)}")
    m.Obj(m.sum(slacks))
    m.options.SOLVER = 1
    m.solver_options = [
        'minlp_print_level 5',
        'max_iter 10000',
        'print_level 5'
    ]
    print("Solving LP...")
    m.solve(disp=True)
    # m.solve(disp=False)
    slack_vals = [float(s.value[0]) for s in slacks]
    violations = sum(1 for v in slack_vals if v > 1e-6)
    return w_d, float(w_m.value[0]), len(slacks), violations
    
def lp_soft_method_pulp(distances, matches, eps=1e-4):   
    Q, N = distances.shape 
    print("using PuLP")
    # Using PuLP
    prob = pulp.LpProblem('VectorRanking', pulp.LpMinimize)
    w_d = 1
    w_m = pulp.LpVariable('w_m', lowBound=0)
    # prob += w_d + w_m == 2 #normalization constraint

    slacks = []
    for q in tqdm(range(Q), desc="Building PuLP constraints"):
        d = distances[q]
        mvals = matches[q]
        pos = np.where(mvals == 1)[0]
        neg = np.where((mvals == 0) | (mvals == 0.5))[0]
        for i in pos:
            # neg_sample = np.random.choice(neg, size=min(10, len(neg)), replace=False)
            for j in neg:
                if d[i] < d[j]:
                    continue
                s = pulp.LpVariable(f's_{q}_{i}_{j}', lowBound=0)
                slacks.append(s)
                prob += (w_d * d[i] + w_m * (1 - mvals[i]) + eps
                            <= w_d * d[j] + w_m * (1 - mvals[j]) + s)
    print(f"Total equations: {len(slacks)}")
    alpha = 500 # or any value you want
    print(f"Using alpha = {alpha} for normalization")

    if len(slacks) > 0:
        avg_slack = pulp.lpSum(slacks) / len(slacks)
    else:
        avg_slack = 0

    prob += w_m + alpha * avg_slack
    # prob += pulp.lpSum(slacks)
    print("Solving LP...")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    slack_vals = [v.value() for v in slacks]
    violations = sum(1 for v in slack_vals if v > 1e-6)
    return w_d, w_m.value(), len(slacks), violations
    
    
def lp_soft_method_without_slack(distances, matches, eps=1e-4, method ='lp_wo_slack'): 
    Q, N = distances.shape   
    print("using PuLP without slacks")
    # Using PuLP
    prob = pulp.LpProblem('VectorRanking', pulp.LpMinimize)
    w_m = pulp.LpVariable('w_m', lowBound=0)
    num_equations = 0
    # prob += w_d + w_m == 2 #normalization constraint

    for q in tqdm(range(Q), desc="Building PuLP constraints"):
        d = distances[q]
        mvals = matches[q]
        pos = np.where(mvals == 1)[0]
        neg = np.where((mvals == 0) | (mvals == 0.5))[0]
        for i in pos:
            for j in neg:
                if d[i] < d[j]:
                    continue
                prob += (d[i] + w_m * (1 - mvals[i]) + eps <= d[j] + w_m * (1 - mvals[j]))
                num_equations += 1
    prob += w_m 
    print("Solving LP...")
    print("eps:", eps)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return w_m.value(), num_equations
    


def main():
    parser = argparse.ArgumentParser(description='Learn weights for vector ranking')
    parser.add_argument('unfiltered_ground_truth', help='Unfiltered Ground truth file (binary format)')
    parser.add_argument('filtered_ground_truth', help='Filtered Ground truth file (binary format)')
    parser.add_argument('filter_matches', help='Filter match file (binary match scores)')
    parser.add_argument('--method', choices=['ratio', 'gekko', 'pulp', 'pulp_wo_slack'], default='ratio')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    # Read the ground truth file
    ground_truth_indices, ground_truth_distances = read_ground_truth(args.unfiltered_ground_truth)
    print("Done reading ground truth file")

    # Read the filter match file
    filter_matches = np.loadtxt(args.filter_matches, dtype=np.int32)
    print(f"Filter matches shape: {filter_matches.shape}")
    print("Done reading filter match file")
    
    # Validate shapes
    if ground_truth_indices.shape != filter_matches.shape:
        print(f"Shape mismatch: {ground_truth_indices.shape} vs {filter_matches.shape}")
        sys.exit(1)
        
    
    # Concatenate filtered and unfiltered ground truth distances and match scores
    # Read filtered ground truth
    filtered_indices, filtered_distances = read_ground_truth(args.filtered_ground_truth)
    # Read unfiltered ground truth (already read as ground_truth_distances)
    # Read filtered match scores (assume first 100 rows from filtered, rest from unfiltered)
    shape = filtered_indices.shape  # or ground_truth_distances.shape
    filter_matches_all = np.ones(shape, dtype=np.int32)
    num_filtered = filtered_indices.shape[0]
    print(f"Number of filtered queries: {num_filtered}")
    
    # Concatenate: first 100 from filtered, rest from unfiltered
    distances = np.concatenate([filtered_distances, ground_truth_distances], axis=1)
    matches = np.concatenate([filter_matches_all, filter_matches_all], axis=1)
    
    print(f"Distances shape: {distances.shape}")
    print(f"Matches shape: {matches.shape}")
    
    print(f"Distances: {distances[0][:5]}")
    # distances_scaled = distances / distances.max()
    # print(f"Scaled distances: {distances_scaled[0][:5]}")
    w_d = 1.0  # Default weight for distances
    violations = 0

    if args.method == 'ratio':
        w_d, w_m, total_pairs, _ = direct_ratio_method(distances, filter_matches, args.eps)
    if args.method == 'pulp_wo_slack':
        w_m, total_pairs = lp_soft_method_without_slack(distances, filter_matches, args.eps)
    if args.method == 'gekko':
        w_d, w_m, total_pairs, violations = lp_soft_method_gekko(distances, filter_matches, args.eps)
    if args.method == 'pulp':
        w_d, w_m, total_pairs, violations = lp_soft_method_pulp(distances, filter_matches, args.eps)

    print(f"Method: {args.method}")
    print(f"w_d = {w_d:.6f}, w_m = {w_m:.6f}")
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"Violations (slack > 0): {violations}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            # Simple bar of satisfied vs violated
            satisfied = total_pairs - violations
            plt.bar(['satisfied', 'violated'], [satisfied, violations])
            plt.title('Constraint Satisfaction')
            plt.ylabel('Count')
            plt.show()
        except ImportError:
            print('matplotlib not installed; cannot plot.')

if __name__ == '__main__':
    main()