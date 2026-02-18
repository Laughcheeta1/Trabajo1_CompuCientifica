import json
import os

input_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1.ipynb"
output_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1_2.ipynb"

# The correct list of columns (with comma)
correct_cols_line = "numerical_cols = ['Year_Birth', 'Income', 'Recency', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Z_CostContact', 'Z_Revenue', 'Response']\n"

try:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixed_cols = False

    # 1. Fix numerical_cols
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source_str = "".join(cell["source"])
            if "numerical_cols =" in source_str:
                # We found the cell.
                # We will replace the source lines that define numerical_cols.
                # Since it might be multi-line, we'll just look for the start and replace the whole block?
                # Or easier: just identify the assignment and overwrite it?
                # The cell seems to contain:
                # # Numerical Distributions
                # numerical_cols = [...]
                # plot_distributions_numerical(...)

                # We'll construct the new source.
                new_source = []
                # Keep comments/imports if any
                for line in cell["source"]:
                    if "numerical_cols =" in line:
                        # Replace this line with the correct one
                        new_source.append(correct_cols_line)
                        fixed_cols = True
                    elif "'NumStorePurchases'" in line:
                        # Skip continuation lines if the previous definition was multi-line
                        # This is tricky without parsing.
                        # Assuming the previous attempt used a list literal that might span lines.
                        # If so, we need to remove the old lines.
                        pass
                    elif "'NumWebVisitsMonth'" in line:
                        pass
                    elif "'Z_Revenue'" in line:  # End of list?
                        pass
                    elif line.strip().startswith("'"):  # Continuation of list items
                        pass
                    else:
                        new_source.append(line)

                # This replacement logic is a bit reduced/risky if we don't know exact multi-line structure.
                # Better approach:
                # If we found "numerical_cols =", we assume it's that cell.
                # We'll just Rewrite the whole cell content since we know what it should be (from previous debug).
                # Cell 7 source was:
                # '# Numerical Distributions\n'
                # "numerical_cols = [...]\n"
                # 'plot_distributions_numerical(df, numerical_cols)'

                cell["source"] = [
                    "# Numerical Distributions\n",
                    correct_cols_line,
                    "plot_distributions_numerical(df, numerical_cols)",
                ]
                fixed_cols = True
                print("Fixed numerical_cols in Cell.")

    # 2. Check for matplotlib import
    # Usually in the first code cell or imports cell.
    # We'll check if 'import matplotlib.pyplot as plt' exists.
    # If not, we append it to the first code cell.

    has_matplotlib = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if "import matplotlib.pyplot" in "".join(cell["source"]):
                has_matplotlib = True
                break

    if not has_matplotlib:
        print("Adding matplotlib import to first code cell.")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                # Prepend import
                cell["source"].insert(0, "import matplotlib.pyplot as plt\n")
                break

    # Write the new notebook
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Created fixed notebook: {output_path}")

except Exception as e:
    print(f"Error: {e}")
