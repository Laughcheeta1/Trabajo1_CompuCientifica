import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame.
    """
    # For now, drop rows with missing values as a simple approach
    # In a real scenario, we might want imputation
    return df.dropna()

def feature_engineering(df):
    """
    Creates new features based on existing columns.
    """
    # Calculate Age
    current_year = 2026
    df['Age'] = current_year - df['Year_Birth']

    # Calculate Tenure
    if 'Dt_Customer' in df.columns:
        # Convert to datetime if not already
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
        # Use a fixed reference date or current date
        reference_date = pd.to_datetime('2026-02-15')
        df['Tenure_Days'] = (reference_date - df['Dt_Customer']).dt.days
    
    # Total Amount Spent
    mnt_cols = [c for c in df.columns if 'Mnt' in c]
    df['Total_Mnt'] = df[mnt_cols].sum(axis=1)
    
    # Total Purchases
    num_cols = [c for c in df.columns if 'Num' in c and 'Purchases' in c]
    df['Total_Num_Purchases'] = df[num_cols].sum(axis=1)
    
    # Family Size
    if 'Marital_Status' in df.columns:
        df['Is_Partner'] = df['Marital_Status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    else:
        df['Is_Partner'] = 0
        
    df['Family_Size'] = df['Is_Partner'] + 1 + df['Kidhome'] + df['Teenhome']
    
    return df

def encode_categorical(df, categorical_cols):
    """
    Encodes categorical columns using Label Encoding.
    """
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(df, numerical_cols):
    """
    Scales numerical features using Standard Scaler.
    Returns a new DataFrame with scaled features and the scaler object.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=df.index)
    return scaled_df, scaler

def preprocess_pipeline(df):
    """
    Runs the full preprocessing pipeline.
    """
    # Check missing values
    df = handle_missing_values(df)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Define columns for encoding
    categorical_cols = ['Education', 'Marital_Status']
    df = encode_categorical(df, categorical_cols)
    
    # Drop non-numerical columns not needed for clustering or kept for reference
    # We keep the ID for reference but won't use it for clustering
    # We drop 'Dt_Customer' as we extracted Tenure
    # We drop 'Year_Birth' as we extracted Age
    
    drop_cols = ['ID', 'Dt_Customer', 'Year_Birth']
    # Also drop categorical original columns if we encoded them? 
    # Label encoding replaces them in place in my function above.
    
    # Select features for clustering
    # We want to use the derived features and relevant original ones
    # Exclude ID, dates, and maybe some original ones if we use totals
    
    clustering_features = [c for c in df.columns if c not in drop_cols and c not in ['Response', 'Complain']] 
    # Response and Complain might be targets or just features, keeping them for now or dropping dependent on goal.
    # The goal is segmentation based on behavior. Response is a result.
    
    return df
