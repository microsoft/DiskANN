import matplotlib.pyplot as plt

# Data for plotting
data = {
    "Baseline": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3687.22, 4195.96, 4685.40, 5171.15, 5654.48, 6129.37],
        "Mean Latency (mus)": [2705.62, 3319.41, 3850.11, 3903.45, 4230.41, 4574.90],
        "Recall@50": [62.96, 65.92, 68.19, 70.07, 71.76, 73.17]
    },
    "R=2": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3434.01, 3901.36, 4359.70, 4799.74, 5237.92, 5669.62],
        "Mean Latency (mus)": [2492.46, 2911.56, 3351.47, 3719.99, 4094.19, 4398.48],
        "Recall@50": [60.19, 63.31, 65.93, 67.98, 69.75, 71.21]
    },
    "KK 1.5": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3481.68, 3960.66, 4428.85, 4895.59, 5347.67, 5802.73],
        "Mean Latency (mus)": [2596.59, 3071.11, 3468.83, 3857.31, 4210.76, 4553.08],
        "Recall@50": [62.22, 65.11, 67.45, 69.45, 71.05, 72.53]
    },
    "KK 3": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [3073.33, 3497.61, 3905.47, 4309.33, 4708.85, 5105.36],
        "Mean Latency (mus)": [2294.82, 2695.71, 3045.03, 3379.80, 3700.46, 4050.05],
        "Recall@50": [58.91, 61.94, 64.37, 66.43, 68.18, 69.70]
    },
    "KK 4": {
        "Ls": [50, 60, 70, 80, 90, 100],
        "Avg dist cmps": [2990.19, 3397.21, 3798.61, 4190.69, 4579.45, 4963.58],
        "Mean Latency (mus)": [2287.41, 2603.87, 2959.77, 3298.64, 3645.76, 3986.09],
        "Recall@50": [56.47, 59.61, 62.18, 64.36, 66.26, 67.88]
    }
}

colors = ['b', 'g', 'r', 'c']
labels = list(data.keys())

# Plot Recall vs Avg dist cmps
plt.figure(figsize=(12, 6))
for i, label in enumerate(labels):
    plt.plot(data[label]["Avg dist cmps"], data[label]["Recall@50"], color=colors[i], marker='o', label=label)
plt.xlabel('Avg dist cmps')
plt.ylabel('Recall@50')
plt.title('Recall vs Avg dist cmps')
plt.legend()
plt.grid(True)
plt.show()

# Plot Recall vs Mean Latency
plt.figure(figsize=(12, 6))
for i, label in enumerate(labels):
    plt.plot(data[label]["Mean Latency (mus)"], data[label]["Recall@50"], color=colors[i], marker='o', label=label)
plt.xlabel('Mean Latency (mus)')
plt.ylabel('Recall@50')
plt.title('Recall vs Mean Latency')
plt.legend()
plt.grid(True)
plt.show()