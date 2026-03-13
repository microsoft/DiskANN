import matplotlib.pyplot as plt

# Data for plotting
data = {
    "Baseline": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3687.22, 4195.96, 4685.40, 5171.15, 5654.48, 6129.37],
        "Mean Latency (mus)": [1332.03, 1503.96, 1678.89, 1852.77, 2027.21, 2204.12],
        "Recall@50": [62.96, 65.92, 68.19, 70.07, 71.76, 73.17]
    },
    "R=2": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3434.01, 3901.36, 4359.70, 4799.74, 5237.92, 5669.62],
        "Mean Latency (mus)": [1239.42, 1407.26, 1573.24, 1734.85, 1898.69, 2053.13],
        "Recall@50": [60.19, 63.31, 65.93, 67.98, 69.75, 71.21]
    },
    "KK 1.5": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3481.68, 3960.66, 4428.85, 4895.59, 5347.67, 5802.73],
        "Mean Latency (mus)": [1265.58, 1442.37, 1616.34, 1789.87, 1956.81, 2124.76],
        "Recall@50": [62.22, 65.11, 67.45, 69.45, 71.05, 72.53]
    },
    "KK 3": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3073.33, 3497.61, 3905.47, 4309.33, 4708.85, 5105.36],
        "Mean Latency (mus)": [1120.41, 1280.12, 1435.12, 1585.94, 1735.12, 1888.07],
        "Recall@50": [58.91, 61.94, 64.37, 66.43, 68.18, 69.70]
    },
    "KK 4": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [2990.19, 3397.21, 3798.61, 4190.69, 4579.45, 4963.58],
        "Mean Latency (mus)": [1094.82, 1244.82, 1402.78, 1552.51, 1698.52, 1842.31],
        "Recall@50": [56.47, 59.61, 62.18, 64.36, 66.26, 67.88]
    }
    ,
    "R 90": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3760.59, 4282.52, 4795.27, 5303.92, 5805.21, 6306.36],
        "Mean Latency (mus)": [1370.57, 1565.07, 1755.34, 1944.71, 2135.67, 2321.62],
        "Recall@50": [63.90, 66.74, 69.00, 70.97, 72.60, 74.05]
    }
}

colors = ['b', 'g', 'r', 'c', 'm', 'y']
labels = list(data.keys())

# Save Recall vs Avg dist cmps plot
plt.figure(figsize=(12, 6))
for i, label in enumerate(labels):
    plt.plot(data[label]["Avg dist cmps"], data[label]["Recall@50"], color=colors[i], marker='o', label=label)
    for j, L in enumerate(data[label]["Ls"]):
        plt.annotate(f'L={L}', (data[label]["Avg dist cmps"][j], data[label]["Recall@50"][j]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Avg dist cmps')
plt.ylabel('Recall@50')
plt.title('Recall vs Avg dist cmps')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_avg_dist_cmps_1_thread.png')

# Save Recall vs Mean Latency plot
plt.figure(figsize=(12, 6))
for i, label in enumerate(labels):
    plt.plot(data[label]["Mean Latency (mus)"], data[label]["Recall@50"], color=colors[i], marker='o', label=label)
    for j, L in enumerate(data[label]["Ls"]):
        plt.annotate(f'L={L}', (data[label]["Mean Latency (mus)"][j], data[label]["Recall@50"][j]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Mean Latency (mus)')
plt.ylabel('Recall@50')
plt.title('Recall vs Mean Latency')
plt.legend()
plt.grid(True)
plt.savefig('recall_vs_mean_latency_1_thread.png')
