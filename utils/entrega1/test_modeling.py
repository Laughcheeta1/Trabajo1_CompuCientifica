from utils.entrega1.data_loader import load_data
from utils.entrega1.preprocessing import preprocess_pipeline, scale_features
from utils.entrega1.modeling import run_kmeans, evaluate_clusters, interpret_clusters
import pandas as pd
import numpy as np

# Load and preprocess
df = load_data('data/datos_caso_1.csv')
if df is not None:
    # Use pipeline mainly for cleaning and feature engineering
    # We need to manually select features or refine pipeline for this test
    # Let's do a quick manual prep based on what the notebook will do
    
    # 1. Pipeline (clean + engineer + encode)
    df_prep = preprocess_pipeline(df)
    
    # 2. Select features for clustering
    # Excluding ID, dates, and dependent variables like Response/Complain/AcceptedCmp
    # Also excluding original unscaled numericals if we use scaled ones? No, scale_features handles scaling.
    # Let's pick a subset for testing: Age, Tenure_Days, Total_Mnt, Family_Size, NumPurchases
    
    cluster_cols = ['Age', 'Tenure_Days', 'Total_Mnt', 'Family_Size'] # + others
    # Ensure these exist
    for c in cluster_cols:
        if c not in df_prep.columns:
            print(f"Column {c} missing")
            
    # 3. Scale
    print("Scaling features...")
    df_scaled, scaler = scale_features(df_prep, cluster_cols)
    print("Scaled shape:", df_scaled.shape)
    
    # 4. Run KMeans
    print("Running KMeans (k=3)...")
    kmeans, labels = run_kmeans(df_scaled, n_clusters=3)
    print("Labels unique:", np.unique(labels))
    
    # 5. Interpret
    print("Interpreting clusters...")
    df_prep['Cluster'] = labels
    summary = interpret_clusters(df_prep, labels)
    print(summary[['Age', 'Total_Mnt']].head())
    
    # 6. Evaluate (mocking plot)
    print("Evaluating clusters (Elbow/Silhouette)...")
    # evaluate_clusters(df_scaled, range(2, 5)) # This would show plot, we just want to ensure it runs
    
else:
    print("Failed to load data.")
