from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import numpy as np

json_file_path = 'English_Songs_Features.json'

# Load features from JSON
with open(json_file_path, 'r') as f:
    features_dict = json.load(f)

# Prepare data for clustering
song_names = list(features_dict.keys())
features = np.array(list(features_dict.values()))

# Apply KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features)
clusters = kmeans.labels_

# Map songs to their clusters
song_clusters = {song_name: cluster for song_name, cluster in zip(song_names, clusters)}

# Perform PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Print song name, PC1, PC2, and cluster for each song
print("Song Name, PC1 (x), PC2 (y), Cluster:")
for i, (x, y) in enumerate(reduced_features):
    print(f"Song: {song_names[i]}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {clusters[i]}")

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis')

# Annotate each song with cluster number
for i, txt in enumerate(clusters):
    plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=9, color='black')

plt.title('English Songs Clustering with Cluster Numbers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()