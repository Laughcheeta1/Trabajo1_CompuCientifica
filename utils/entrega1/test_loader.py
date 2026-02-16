from utils.entrega1.data_loader import load_data
import os

# Define path relative to where this script will be run (usually project root)
data_path = 'data/datos_caso_1.csv'

# Test loading
df = load_data(data_path)

if df is not None:
    print("Head of the data:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
else:
    print("Failed to load data.")
