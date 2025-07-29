import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def get_neutrailty_score(vector):
    return np.average(vector)*np.var(vector)

def reduce_dims(vectors, dims):
    pca = PCA(n_components=dims)
    new_vectors = pca.fit_transform(vectors)
    return new_vectors

def cluster_dimensions(vectors, clusters = 10):
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(vectors)
    groups = []

    for i in range(clusters):
        groups.append(np.where(kmeans.labels_ == i)[0])
    for i in range(clusters):
        closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
    return closests