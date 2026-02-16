from utils.entrega1.data_loader import load_data
from utils.entrega1.preprocessing import preprocess_pipeline, handle_missing_values, feature_engineering, encode_categorical
import pandas as pd

# Load data
df = load_data('data/datos_caso_1.csv')

if df is not None:
    print("Original Shape:", df.shape)
    
    # Test individual steps
    df_clean = handle_missing_values(df.copy())
    print("Shape after dropping missing:", df_clean.shape)
    
    df_feat = feature_engineering(df_clean.copy())
    print("Columns after feature engineering:", df_feat.columns)
    print("Head of new features:\n", df_feat[['Age', 'Tenure_Days', 'Total_Mnt', 'Family_Size']].head())

    # Test full pipeline
    # Note: pipeline as written returns the df but doesn't drop all columns yet, need to refine or just use steps manually in notebook
    # The function preprocess_pipeline in preprocessing.py returns the df but we might want to select specific columns for clustering
    # Let's adjust preprocessing.py if needed or just use the helper functions. 
    # For now, let's just see if functions run.
    
    print("\nEncoding categorical...")
    print(df_feat['Education'].unique())
    df_encoded = encode_categorical(df_feat.copy(), ['Education', 'Marital_Status'])
    print("Encoded Education unique values:", df_encoded['Education'].unique())

else:
    print("Failed to load data.")
