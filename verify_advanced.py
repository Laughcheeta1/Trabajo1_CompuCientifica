
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulate display function
def display(obj):
    print(obj)

# Ensure utils can be imported
sys.path.append(os.path.abspath('.'))

from utils.entrega1.data_loader import load_data
from utils.entrega1.preprocessing import preprocess_pipeline, scale_features
from utils.entrega1.modeling import (run_clustering_models, compare_models, evaluate_clusters_kmeans, 
                                     plot_dendrogram, plot_knn_distance, optimize_dbscan_grid, run_dbscan)
from sklearn.cluster import AgglomerativeClustering

# Load Data
print("Loading Data...")
df = load_data('data/datos_caso_1.csv')
if df is None:
    sys.exit(1)

# Preprocess
print("Preprocessing...")
df_prep = preprocess_pipeline(df)
features_to_cluster = ['Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 
                       'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
                       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                       'NumWebVisitsMonth', 'Tenure_Days', 'Total_Mnt', 'Total_Num_Purchases', 
                       'Family_Size']
features_to_cluster = [c for c in features_to_cluster if c in df_prep.columns]
df_scaled, scaler = scale_features(df_prep, features_to_cluster)

# 1. K-Means
print("\n--- Testing K-Means ---")
evaluate_clusters_kmeans(df_scaled, range(2, 4))

# 2. Hierarchical
print("\n--- Testing Hierarchical (Dendrogram) ---")
model_avg = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average')
model_avg = model_avg.fit(df_scaled)
plot_dendrogram(model_avg, truncate_mode='level', p=3)
# plt.show() # Mock

# 3. DBSCAN
print("\n--- Testing DBSCAN Optimization ---")
# KNN
plot_knn_distance(df_scaled, k=5)

# Grid Search
eps_range = np.arange(1.5, 2.0, 0.5) # Small range for test
min_samples_range = range(5, 10, 5)
dbscan_results = optimize_dbscan_grid(df_scaled, eps_range, min_samples_range)
print("DBSCAN Results Head:")
print(dbscan_results.head())

if not dbscan_results.empty:
    best_row = dbscan_results.loc[dbscan_results['Score'].idxmax()]
    print(f"Best: Eps={best_row['Epsilon']}, MinSamples={best_row['MinSamples']}")

# 4. Comparison
print("\n--- Testing Comparison ---")
results = run_clustering_models(df_scaled, n_clusters=3)
results['DBSCAN'] = run_dbscan(df_scaled, eps=1.5, min_samples=5)
metrics = compare_models(df_scaled, results)
print("Metrics:")
print(metrics)

print("\n--- Verification Complete ---")
