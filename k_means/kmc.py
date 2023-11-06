import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=10):
        self.k = k
        self.max_iters = max_iters

    def do_cluster(self, data):
        self.data = data
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            self.labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2), axis=1)
            
            # Update centroids by taking the mean of all data points in each cluster
            new_centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
        return (self.centroids,self.labels)

    def plot_clusters(self):
        # Plot data points with different colors for each cluster
        plt.figure(figsize=(8, 6))
        for i in range(self.k):
            cluster_data = self.data[self.labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}')

        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='^', color='black', s=200, label='Centroids')
        
        plt.title('K-Means Clustering')
        plt.legend()
        plt.show()

# if __name__ == "__main__":
#     # Generate some random data for demonstration
#     np.random.seed(0)
#     data = np.random.rand(100, 2)
    
#     # Specify the number of clusters (k)
#     k = 3
    
#     # Create a KMeans object and perform clustering
#     kmeans = KMeans(k)
#     kmeans.do_cluster(data)
    
#     # Visualize the clusters
#     kmeans.plot_clusters()
