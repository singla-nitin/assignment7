import numpy as np
from sklearn.metrics import pairwise_distances

def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    centroids = X[indices[:k]]
    return centroids

def assign_clusters(X, centroids):
    distances = pairwise_distances(X, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)

def update_centroids(X, clusters, k):
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def calculate_sse(X, clusters, centroids):
    sse = 0
    for i in range(centroids.shape[0]):
        cluster_points = X[clusters == i]
        sse += np.sum((cluster_points - centroids[i]) ** 2)
    return sse

def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        # If centroids don't change, convergence has occurred
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    sse = calculate_sse(X, clusters, centroids)
    return clusters, centroids, sse

def elbow_method(X, max_k):
    sse_values = []
    for k in range(1, max_k+1):
        _, _, sse = k_means(X, k)
        sse_values.append(sse)

    plt.plot(range(1, max_k+1), sse_values, marker='o')
    plt.title("Elbow Method for Optimal K (K-Means)")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("SSE")
    plt.show()

elbow_method(X_scaled, max_k=10)
