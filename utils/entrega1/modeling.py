from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# Requires scikit-learn-extra
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print("scikit-learn-extra not installed. KMedoids will not work.")
    KMedoids = None

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def run_clustering_models(X, n_clusters=3, random_state=42):
    """
    Runs multiple clustering algorithms and returns their labels and metrics.
    Algorithms: KMeans, KMedoids, Agglomerative, GMM.
    DBSCAN is handled separately as it determines clusters automatically.
    """
    results = {}
    
    # 1. KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels_kmeans = kmeans.fit_predict(X)
    results['KMeans'] = labels_kmeans
    
    # 2. KMedoids
    if KMedoids:
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state)
        labels_kmedoids = kmedoids.fit_predict(X)
        results['KMedoids'] = labels_kmedoids
        
    # 3. Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X)
    results['Agglomerative'] = labels_agg
    
    # 4. GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels_gmm = gmm.fit_predict(X)
    results['GMM'] = labels_gmm
    
    return results

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def compare_models(X, results_dict):
    """
    Calculates and plots metrics for different models.
    Metrics: Silhouette Score, Davies-Bouldin Index.
    """
    metrics = []
    
    for name, labels in results_dict.items():
        # Filter out noise for DBSCAN if present (-1)
        if len(set(labels)) > 1:
            try:
                sil = silhouette_score(X, labels)
                db = davies_bouldin_score(X, labels)
                metrics.append({'Model': name, 'Silhouette': sil, 'Davies-Bouldin': db})
            except ValueError:
                print(f"Could not calculate metrics for {name}")
                
    if not metrics:
        print("No valid models to compare.")
        return

    metrics_df = pd.DataFrame(metrics)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x='Model', y='Silhouette', data=metrics_df, ax=axes[0], hue='Model', palette='viridis', legend=False)
    axes[0].set_title('Silhouette Score (Higher is better)')
    
    sns.barplot(x='Model', y='Davies-Bouldin', data=metrics_df, ax=axes[1], hue='Model', palette='magma', legend=False)
    axes[1].set_title('Davies-Bouldin Index (Lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

def evaluate_clusters_kmeans(X, range_n_clusters):
    """
    Evaluates K-Means for different k (Elbow & Silhouette).
    """
    inertias = []
    silhouette_scores = []
    
    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None)
            
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(range_n_clusters, inertias, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)  
    ax2.plot(range_n_clusters, silhouette_scores, 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('K-Means Optimization: Elbow & Silhouette')
    plt.tight_layout()
    plt.show()

def visualize_clusters_pca(X, labels, title='Clusters Visualization (PCA)'):
    """
    Visualizes clusters using PCA (2D).
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, style=labels)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def interpret_clusters(df, labels):
    """
    Returns mean values for numeric features by cluster.
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    # Select only numeric columns for groupby mean to avoid errors with new pandas versions
    numeric_cols = df_clustered.select_dtypes(include=['number']).columns
    return df_clustered.groupby('Cluster')[numeric_cols].mean()

def plot_dendrogram(model, **kwargs):
    """
    Plots a dendrogram for an AgglomerativeClustering model.
    Borrowed from sklearn examples.
    """
    from scipy.cluster.hierarchy import dendrogram
    
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_knn_distance(X, k=5):
    """
    Plots the k-nearest neighbors distance to help Estimate Epsilon for DBSCAN.
    """
    from sklearn.neighbors import NearestNeighbors
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort distance values by ascending value and plot
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Nearest Neighbors Distance (k={k})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Epsilon (Distance to k-th neighbor)')
    plt.grid(True)
    plt.show()

def optimize_dbscan_grid(X, eps_values, min_samples_values):
    """
    Grid search for DBSCAN hyperparameters and visualizes Silhouette scores in a heatmap.
    """
    from itertools import product
    
    dbscan_params = list(product(eps_values, min_samples_values))
    sil_scores = []
    
    for p in dbscan_params:
        try:
            y_pred = DBSCAN(eps=p[0], min_samples=p[1]).fit_predict(X)
            # Silhouette requires more than 1 cluster and less than N samples-1
            unique_labels = set(y_pred)
            if len(unique_labels) > 1 and len(unique_labels) < len(X):
                score = silhouette_score(X, y_pred)
            else:
                score = -1 # Bad score for noise-only or single-cluster results
            sil_scores.append(score)
        except Exception:
            sil_scores.append(-1)

    df_param_adj = pd.DataFrame.from_records(dbscan_params, columns=['Epsilon', 'MinSamples'])
    df_param_adj['Score'] = sil_scores

    pivot_data = pd.pivot_table(df_param_adj, values='Score', index='MinSamples', columns='Epsilon')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('DBSCAN Silhouette Score Heatmap')
    plt.show()
    
    return df_param_adj
