import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data preparation
topk_data = pd.DataFrame({
    'metric': ['random', 'random', 'random', 'raw', 'raw', 'raw', 'VAE', 'VAE', 'VAE', 'DEC', 'DEC', 'DEC'],
    'topk':    [1, 2, 3] * 4,
    'accuracy':[5.3, 5.1, 4.2, 8.4, 8.7, 7.2, 12.2, 10.8, 9.0, 8.4, 8.7, 8.5]
})

clustering_data = pd.DataFrame({
    'method': ['louvain', 'louvain', 'infomap', 'infomap'],
    'KNN': [3, 10, 3, 10],
    'accuracy': [7.5, 6.3, 6.4, 4.2]
})

sim_data = pd.DataFrame({
    'metric': ['random', 'raw', 'VAE'],
    'Bottom_Acc': [5.3, 1.6, 1.3],
    'Top_Acc': [5.3, 8.4, 12.2]
})

# Compute mean accuracy per metric
mean_data = topk_data.groupby('metric')['accuracy'].mean()

# Define color palette for K values and average
k_colors = {1: 'gold', 2: 'orange', 3: 'red', 'avg': 'purple'}

# Plotting with adjusted width ratios
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[0.65, 0.35])

# Left large plot: Top-K Recommendation Accuracy with average
ax1 = fig.add_subplot(gs[:, 0])
metrics = topk_data['metric'].unique()
x = np.arange(len(metrics))
width = 0.15
offsets = {1: -1.5*width, 2: -0.5*width, 3: 0.5*width, 'avg': 1.5*width}

for k, color in k_colors.items():
    if k == 'avg':
        values = [mean_data[m] for m in metrics]
        label = 'Avg'
    else:
        subset = topk_data[topk_data['topk'] == k]
        values = [subset[subset['metric'] == m]['accuracy'].values[0] for m in metrics]
        label = f'K={k}'
    ax1.bar(x + offsets[k], values, width, label=label, color=color)

ax1.set_title('Top-K Recommendation Accuracy')
ax1.set_xlabel('Metric')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(title='Legend')

# Right top: Clustering Accuracy heatmap
ax2 = fig.add_subplot(gs[0, 1])
pivot = clustering_data.pivot(index='method', columns='KNN', values='accuracy')
im = ax2.imshow(pivot, aspect='auto')
ax2.set_title('Clustering Accuracy')
ax2.set_xlabel('KNN')
ax2.set_ylabel('Method')
ax2.set_xticks(np.arange(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns)
ax2.set_yticks(np.arange(len(pivot.index)))
ax2.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax2.text(j, i, f"{pivot.iat[i, j]:.1f}", ha='center', va='center')

# Right bottom: Bottom vs Top Sim Accuracy (Dumbbell plot)
ax3 = fig.add_subplot(gs[1, 1])
y = np.arange(len(sim_data))
ax3.hlines(y, sim_data['Bottom_Acc'], sim_data['Top_Acc'])
ax3.scatter(sim_data['Bottom_Acc'], y, label='Bottom', marker='o')
ax3.scatter(sim_data['Top_Acc'], y, label='Top', marker='o')
ax3.set_yticks(y)
ax3.set_yticklabels(sim_data['metric'])
ax3.set_title('Bottom vs Top Sim Accuracy')
ax3.set_xlabel('Accuracy (%)')
ax3.legend()

plt.show()
