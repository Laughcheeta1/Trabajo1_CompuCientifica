from utils.entrega1.data_loader import load_data
from utils.entrega1.eda import get_basic_stats, check_missing_values
import os

# Define path relative to where this script will be run (usually project root)
data_path = 'data/datos_caso_1.csv'

# Test loading
df = load_data(data_path)

if df is not None:
    print("Basic Stats Head:")
    print(get_basic_stats(df).head())
    
    print("\nMissing Values:")
    print(check_missing_values(df))
    
    # We won't test plots interactively here, but we can verify the function calls don't crash if we mock show()
    # Or just rely on visual verification in the notebook later. for now just testing stats.
else:
    print("Failed to load data for EDA test.")
