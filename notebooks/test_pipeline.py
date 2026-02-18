"""
Test script for the full clustering pipeline.
Run this to verify everything works before converting to a notebook.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Use non-interactive backend so plots don't block execution
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure utils can be found
sys.path.append(os.path.abspath(".."))

# ====================================================================
# SECTION 1: Data Loading
# ====================================================================
print("=" * 60)
print("SECTION 1: Data Loading")
print("=" * 60)

from utils.entrega1.data_loader import load_data

df = load_data("../data/datos_caso_1.csv")
print(f"Shape: {df.shape}")
print(df.dtypes)

# ====================================================================
# SECTION 2: EDA
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 2: EDA")
print("=" * 60)

from utils.entrega1.eda import (
    get_basic_stats,
    check_missing_values_viz,
    plot_distributions_numerical,
    plot_boxplots,
    plot_pie_categorical,
    plot_correlation_matrix,
)

# Basic stats
stats = get_basic_stats(df)
print("Basic stats OK:", stats.shape)

# Missing values viz
check_missing_values_viz(df)
print("Missing values viz OK")

# Numerical columns for distribution and boxplots
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

plot_distributions_numerical(df, numerical_cols)
print("Distributions OK")

plot_boxplots(df, numerical_cols)
print("Boxplots OK")

# Categorical columns for pie charts
categorical_cols_eda = ["Education", "Marital_Status"]
plot_pie_categorical(df, categorical_cols_eda)
print("Pie charts OK")

# Correlation matrix
plot_correlation_matrix(df)
print("Correlation matrix OK")

# ====================================================================
# SECTION 3: Preprocessing
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 3: Preprocessing")
print("=" * 60)

from utils.entrega1.preprocessing import preprocess_pipeline

(
    df_scaled,
    scaler,
    encoder,
    all_feature_cols,
    num_feature_cols,
    ohe_feature_cols,
    categorical_cols,
) = preprocess_pipeline(df)

print(f"df_scaled shape: {df_scaled.shape}")
print(f"all_feature_cols ({len(all_feature_cols)}): {all_feature_cols}")
print(f"num_feature_cols ({len(num_feature_cols)}): {num_feature_cols}")
print(f"ohe_feature_cols ({len(ohe_feature_cols)}): {ohe_feature_cols}")
print(f"categorical_cols: {categorical_cols}")
print(f"Scaler type: {type(scaler)}")
print(f"Encoder type: {type(encoder)}")

# ====================================================================
# SECTION 4: K-Means
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 4: K-Means")
print("=" * 60)

from sklearn.cluster import KMeans
from utils.entrega1.modeling import evaluate_clusters_kmeans

# Elbow + Silhouette (range_n_clusters must be a range or list)
evaluate_clusters_kmeans(
    df_scaled,
    range_n_clusters=range(2, 11),
    include_silhouette=True,
    ref_cluster=3,
)
print("K-Means evaluation OK")

# Train final model
n_clusters_kmeans = 3
kmeans_final = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(df_scaled)
print(
    f"K-Means labels distribution:\n{pd.Series(kmeans_labels).value_counts().sort_index()}"
)

# ====================================================================
# SECTION 5: K-Means Interpretation (Inverse Transform)
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 5: K-Means Cluster Interpretation")
print("=" * 60)

# Get cluster centers
centers_scaled = pd.DataFrame(kmeans_final.cluster_centers_, columns=all_feature_cols)
print(f"Centers scaled shape: {centers_scaled.shape}")

# Inverse transform numerical features
centers_num_inverse = scaler.inverse_transform(centers_scaled[num_feature_cols])
centers_num_df = pd.DataFrame(centers_num_inverse, columns=num_feature_cols)
print(f"Numerical centers shape: {centers_num_df.shape}")

# Inverse transform categorical features
centers_cat_inverse = encoder.inverse_transform(centers_scaled[ohe_feature_cols])
centers_cat_df = pd.DataFrame(centers_cat_inverse, columns=categorical_cols)
print(f"Categorical centers shape: {centers_cat_df.shape}")

# Combine
centers_original = pd.concat([centers_num_df, centers_cat_df], axis=1)
print(f"Combined centers shape: {centers_original.shape}")
print("Cluster centers (original scale):")
print(centers_original.T)

# ====================================================================
# SECTION 6: Hierarchical Clustering (MJA)
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 6: Hierarchical Clustering (MJA)")
print("=" * 60)

from sklearn.cluster import AgglomerativeClustering
from utils.entrega1.modeling import plot_dendrogram

# For dendrogram, we need distance_threshold=0 and n_clusters=None
# to compute the full tree, then plot_dendrogram uses model.children_ etc.
hierarchical_full = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None,
    linkage="ward",
)
hierarchical_full.fit(df_scaled)
print("Full hierarchical tree fitted OK")

plt.figure(figsize=(14, 8))
plt.title("Dendrograma - Método Jerárquico Aglomerativo")
plot_dendrogram(hierarchical_full, truncate_mode="level", p=5)
plt.xlabel("Índice de la muestra (o tamaño del cluster)")
plt.savefig("_test_dendrogram.png")
plt.close()
print("Dendrogram plotted OK")

# Train with specific n_clusters
n_clusters_hierarchical = 3
hierarchical = AgglomerativeClustering(
    n_clusters=n_clusters_hierarchical,
    linkage="ward",
)
hierarchical_labels = hierarchical.fit_predict(df_scaled)
print(
    f"Hierarchical labels distribution:\n{pd.Series(hierarchical_labels).value_counts().sort_index()}"
)

# ====================================================================
# SECTION 7: DBSCAN
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 7: DBSCAN")
print("=" * 60)

from sklearn.cluster import DBSCAN
from utils.entrega1.modeling import plot_knn_distance, optimize_dbscan_grid

# KNN distance plot
plot_knn_distance(df_scaled, k=5)
print("KNN distance plot OK")

# Grid search
dbscan_results_df = optimize_dbscan_grid(
    df_scaled,
    eps_values=np.arange(1.0, 5.0, 0.5),
    min_samples_values=[3, 5, 7, 10],
)
print(f"DBSCAN grid search results (type={type(dbscan_results_df)}):")
print(dbscan_results_df.head(10))

# Pick best params from results
best_row = dbscan_results_df.loc[dbscan_results_df["Score"].idxmax()]
best_eps = best_row["Epsilon"]
best_min_samples = int(best_row["Vecindad"])
print(
    f"Best eps={best_eps}, min_samples={best_min_samples}, score={best_row['Score']:.4f}"
)

# Train DBSCAN
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(df_scaled)
print(
    f"DBSCAN labels distribution:\n{pd.Series(dbscan_labels).value_counts().sort_index()}"
)
print(f"Noise points (label=-1): {(dbscan_labels == -1).sum()}")

# ====================================================================
# SECTION 8: GMM
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 8: GMM")
print("=" * 60)

from sklearn.mixture import GaussianMixture
from utils.entrega1.modeling import evaluate_gmm_bic

# BIC evaluation (n_components_range must be a range or list)
evaluate_gmm_bic(
    df_scaled,
    n_components_range=range(2, 11),
)
print("GMM BIC evaluation OK")

# Train GMM
n_components_gmm = 3
gmm = GaussianMixture(
    n_components=n_components_gmm,
    covariance_type="full",
    random_state=42,
)
gmm_labels = gmm.fit_predict(df_scaled)
print(f"GMM labels distribution:\n{pd.Series(gmm_labels).value_counts().sort_index()}")

# ====================================================================
# SECTION 9: Model Comparison
# ====================================================================
print("\n" + "=" * 60)
print("SECTION 9: Model Comparison")
print("=" * 60)

from utils.entrega1.modeling import (
    compare_all_models_silhouette,
    visualize_clusters_pca,
)

models_dict = {
    "K-Means": kmeans_labels,
    "Jerárquico": hierarchical_labels,
    "DBSCAN": dbscan_labels,
    "GMM": gmm_labels,
}

scores = compare_all_models_silhouette(df_scaled, models_dict)
print(f"Scores dict: {scores}")

# PCA visualizations
for name, labels in models_dict.items():
    visualize_clusters_pca(df_scaled, labels, title=f"{name} Clusters (PCA)")
    print(f"PCA viz for {name} OK")

print("\n" + "=" * 60)
print("ALL SECTIONS PASSED SUCCESSFULLY!")
print("=" * 60)
