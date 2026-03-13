import matplotlib.pyplot as plt

# Data for different configurations
configurations = [
    {
        'label': 'baseline',
        'recall': [80.60, 88.66, 92.12, 94.05, 95.26, 96.09],
        'QPS': [204.00, 122.17, 90.58, 73.30, 62.50, 55.09],
        'avg_dist_cmps': [7716.33, 13663.56, 19477.48, 25174.82, 30768.75, 36281.84],
        'avg_hops': [59.49, 108.56, 158.08, 207.77, 257.52, 307.35],
        'mean_latency': [4901.96, 8185.00, 11039.73, 13642.69, 15998.88, 18151.68]
    },
    {
        'label': 'x=5',
        'recall': [75.56, 82.88, 86.48, 88.72, 90.28, 91.43],
        'QPS': [138.48, 84.11, 68.15, 58.07, 49.56, 44.64],
        'avg_dist_cmps': [5826.41, 8608.48, 10925.24, 13030.90, 14996.30, 16868.74],
        'avg_hops': [44.82, 66.84, 85.45, 102.54, 118.66, 134.20],
        'mean_latency': [7221.22, 11889.23, 14674.08, 17219.57, 20178.49, 22402.45]
    },
    {
        'label': 'x=7',
        'recall': [78.15, 85.50, 88.87, 90.86, 92.18, 93.15],
        'QPS': [219.83, 138.20, 105.45, 86.58, 74.60, 66.05],
        'avg_dist_cmps': [6571.48, 10162.82, 13149.50, 15806.06, 18226.08, 20523.22],
        'avg_hops': [50.53, 79.23, 103.48, 125.31, 145.43, 164.73],
        'mean_latency': [4549.00, 7236.04, 9482.81, 11550.14, 13404.36, 15140.81]
    },
    {
        'label': 'x=10',
        'recall': [79.69, 87.33, 90.61, 92.49, 93.71, 94.55],
        'QPS': [198.75, 118.69, 88.00, 71.21, 60.84, 53.51],
        'avg_dist_cmps': [7149.92, 11640.27, 15499.88, 18942.36, 22119.38, 25015.15],
        'avg_hops': [55.00, 91.23, 122.88, 151.51, 178.28, 202.91],
        'mean_latency': [5031.34, 8425.49, 11363.82, 14041.94, 16437.29, 18686.43]
    }
    # Add more configurations here
]

# Colors for different configurations
colors = ['b', 'g', 'r', 'y']

# Plot latency vs recall
plt.figure()
for i, config in enumerate(configurations):
    plt.plot(config['mean_latency'], config['recall'], marker='o', color=colors[i % len(colors)], label=config['label'])
plt.xlabel('Mean Latency (mus)')
plt.ylabel('Recall@50')
plt.title('Mean Latency vs Recall')
plt.legend()
plt.grid(True)
plt.savefig('prec40M/latency_vs_recall.png')

# Plot QPS vs recall
plt.figure()
for i, config in enumerate(configurations):
    plt.plot(config['QPS'], config['recall'], marker='o', color=colors[i % len(colors)], label=config['label'])
plt.xlabel('QPS')
plt.ylabel('Recall@50')
plt.title('QPS vs Recall')
plt.legend()
plt.grid(True)
plt.savefig('prec40M/qps_vs_recall.png')

# Plot hops vs recall
plt.figure()
for i, config in enumerate(configurations):
    plt.plot(config['avg_hops'], config['recall'], marker='o', color=colors[i % len(colors)], label=config['label'])
plt.xlabel('Avg Hops')
plt.ylabel('Recall@50')
plt.title('Avg Hops vs Recall')
plt.legend()
plt.grid(True)
plt.savefig('prec40M/hops_vs_recall.png')

# Plot dist comps vs recall
plt.figure()
for i, config in enumerate(configurations):
    plt.plot(config['avg_dist_cmps'], config['recall'], marker='o', color=colors[i % len(colors)], label=config['label'])
plt.xlabel('Avg Dist Comps')
plt.ylabel('Recall@50')
plt.title('Avg Dist Comps vs Recall')
plt.legend()
plt.grid(True)
plt.savefig('prec40M/dist_comps_vs_recall.png')
