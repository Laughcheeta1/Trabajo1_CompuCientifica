import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    df["Age"] = current_year - df["Year_Birth"]

    # Calculate Days Being Customer (using current timestamp)
    if "Dt_Customer" in df.columns:
        # Convert to datetime if not already
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
        df["Days_Being_Customer"] = df["Dt_Customer"].apply(
            lambda x: (pd.Timestamp.now() - x).days
        )

    # Total Amount Spent
    mnt_cols = [c for c in df.columns if "Mnt" in c]
    df["Total_Mnt"] = df[mnt_cols].sum(axis=1)

    # Total Purchases
    num_cols = [c for c in df.columns if "Num" in c and "Purchases" in c]
    df["Total_Num_Purchases"] = df[num_cols].sum(axis=1)

    # Partner indicator
    if "Marital_Status" in df.columns:
        df["Is_Partner"] = df["Marital_Status"].apply(
            lambda x: 1 if x in ["Married", "Together"] else 0
        )
    else:
        df["Is_Partner"] = 0

    # Total Offers Accepted
    df["Total_Offers_Accepted"] = (
        df["AcceptedCmp1"]
        + df["AcceptedCmp2"]
        + df["AcceptedCmp3"]
        + df["AcceptedCmp4"]
        + df["AcceptedCmp5"]
        + df["Response"]
    )

    return df


def encode_categorical(df, categorical_cols):
    """
    Encodes categorical columns using OneHotEncoder.
    """
    # Create the encoder
    # sparse_output=False ensures we get a dense array, compatible with pandas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Fit and transform the data
    encoded_data = encoder.fit_transform(df[categorical_cols])

    # Create a DataFrame with the encoded data
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

    # Concatenate with the original DataFrame and drop original columns
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=categorical_cols)

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


def remove_atypical_values(df):
    """
    Removes atypical values from the DataFrame.
    """
    df = df[df["Year_Birth"] > 1920]

    df = df[~df["Marital_Status"].isin(["Absurd", "YOLO"])]

    df = df[df["Income"] < 200000]

    df = df[df["MntMeatProducts"] < 1250]

    return df


def transform_atypical_values(df):
    df["Marital_Status"] = df["Marital_Status"].replace("Alone", "Single")

    return df


def preprocess_pipeline(df):
    """
    Runs the full preprocessing pipeline.

    Returns:
    --------
    tuple : (df_scaled, scaler, encoder, all_feature_cols, num_feature_cols, ohe_feature_cols, categorical_cols)
        - df_scaled: DataFrame with scaled features (numerical + OHE, all scaled together)
        - scaler: Fitted StandardScaler object (fitted on numerical features only)
        - encoder: Fitted OneHotEncoder object
        - all_feature_cols: List of all feature column names (numerical + OHE)
        - num_feature_cols: List of numerical feature column names (before OHE)
        - ohe_feature_cols: List of OHE column names
        - categorical_cols: List of original categorical column names
    """
    # Check missing values
    df = handle_missing_values(df)

    # Remove atypical valuess
    df = remove_atypical_values(df)

    # Transform atypical values
    df = transform_atypical_values(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Define columns for encoding
    categorical_cols = ["Education", "Marital_Status"]

    # Store encoder before encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[categorical_cols])

    df = encode_categorical(df, categorical_cols)

    # Drop non-numerical columns not needed for clustering or kept for reference
    drop_cols = ["ID", "Dt_Customer", "Year_Birth"]
    # Provide features for clustering
    # We filter out non-numeric and excluded columns

    # First drop the explicit drop columns
    df_for_clustering = df.drop(columns=drop_cols, errors="ignore")

    # Also exclude Response and Complain if they are not features
    exclude_cols = ["Response", "Complain"]
    all_clustering_features = [
        c
        for c in df_for_clustering.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_for_clustering[c])
    ]

    # Separate OHE columns from numerical columns
    ohe_cols = [
        c
        for c in all_clustering_features
        if c.startswith("Education_") or c.startswith("Marital_Status_")
    ]
    num_cols = [c for c in all_clustering_features if c not in ohe_cols]

    # Scale ONLY numerical features
    df_num_scaled, scaler = scale_features(df_for_clustering, num_cols)

    # Get OHE features (already 0/1, no scaling needed but we'll include them)
    df_ohe = df_for_clustering[ohe_cols]

    # Combine scaled numerical + OHE features
    df_scaled = pd.concat([df_num_scaled, df_ohe], axis=1)

    # Return scaled dataframe and transformers for inverse transformation
    return (
        df_scaled,
        scaler,
        encoder,
        all_clustering_features,
        num_cols,
        ohe_cols,
        categorical_cols,
    )
