import matplotlib.pyplot as plt

# Extracted data for config1
config1_distances = [3395.25, 3863.55, 4326.47, 4776.06, 5222.84, 5664.97]
config1_recall = [61.76, 64.67, 67.07, 68.99, 70.64, 72.06]

# Extracted data for config2
config2_distances = [3687.22, 4195.96, 4685.40, 5171.15, 5654.48, 6129.37]
config2_recall = [62.96, 65.92, 68.19, 70.07, 71.76, 73.17]

plt.figure(figsize=(10, 6))

# Plot for config1
plt.plot(config1_distances, config1_recall, 'o-', label='ReducePool')

# Plot for config2
plt.plot(config2_distances, config2_recall, 'o--', label='Baseline')

plt.xlabel('Distance Comparisons')
plt.ylabel('Recall')
plt.title('Distance Comparisons vs Recall')
plt.legend()
plt.grid(True)
plt.savefig('distance_recall_plot.png')