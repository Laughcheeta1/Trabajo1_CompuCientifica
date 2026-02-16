
import sys
import os
import matplotlib.pyplot as plt

# Simulate display function
def display(obj):
    print(obj)

# Ensure utils can be imported
sys.path.append(os.path.abspath('.'))

# --- Cell 1: Imports ---
from utils.entrega1.data_loader import load_data
from utils.entrega1.eda import get_basic_stats, check_missing_values_viz, plot_distributions_numerical, plot_boxplots, plot_pie_categorical, plot_correlation_matrix
from utils.entrega1.preprocessing import preprocess_pipeline, scale_features
# FIX: Added run_dbscan to import
from utils.entrega1.modeling import run_clustering_models, compare_models, evaluate_clusters_kmeans, visualize_clusters_pca, interpret_clusters, run_dbscan

# --- Cell 2: Load Data ---
print("Loading Data...")
# Check if data file exists relative to where we run this script
data_file = 'data/datos_caso_1.csv'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found")
    sys.exit(1)

df = load_data(data_file)
if df is None:
    print("Error loading data")
    sys.exit(1)

df.info()

# --- Cell 3: EDA ---
print("\n--- EDA ---")
print("Basic Stats Head:")
display(get_basic_stats(df).T.head())

# Skipping visual plots for script execution speed/suppression unless needed
# check_missing_values_viz(df) 
# plot_distributions_numerical(...)
# plot_boxplots(...)
# plot_pie_categorical(...)
# plot_correlation_matrix(...)

# --- Cell 4: Preprocessing ---
print("\n--- Preprocessing ---")
df_prep = preprocess_pipeline(df)
print("Preprocessed shape:", df_prep.shape)

features_to_cluster = ['Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 
                       'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
                       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                       'NumWebVisitsMonth', 'Tenure_Days', 'Total_Mnt', 'Total_Num_Purchases', 
                       'Family_Size']

features_to_cluster = [c for c in features_to_cluster if c in df_prep.columns]
print(f"Features selected: {len(features_to_cluster)}")

df_scaled, scaler = scale_features(df_prep, features_to_cluster)
print("Scaled data head:")
print(df_scaled.head())

# --- Cell 5: Modeling ---
print("\n--- Modeling ---")
print("Running and Comparing Models (k=3 where applicable)...")
results = run_clustering_models(df_scaled, n_clusters=3)

# Evaluate DBSCAN separately
# Note: DBSCAN often returns -1 for noise.
print("Running DBSCAN...")
labels_dbscan = run_dbscan(df_scaled, eps=1.5, min_samples=10) 
results['DBSCAN'] = labels_dbscan

# Visualize Comparison Metrics
print("Comparing Models...")
# Mocking plot show
metrics_df = compare_models(df_scaled, results)
display(metrics_df)

# --- Cell 6: KMeans Deep Dive ---
print("\n--- KMeans Deep Dive ---")
# evaluate_clusters_kmeans(df_scaled, range(2, 5)) # reduced range for speed

# --- Cell 7: Interpretation ---
print("\n--- Interpretation ---")
# visualize_clusters_pca(df_scaled, results['KMeans'], title='K-Means Clusters Visualization (k=3)')

print("Cluster Interpretation (Mean Values):")
if 'KMeans' in results:
    cluster_summary = interpret_clusters(df_prep, results['KMeans'])
    display(cluster_summary[features_to_cluster].T)
else:
    print("KMeans not in results")

print("\n--- Script Finished Successfully ---")
