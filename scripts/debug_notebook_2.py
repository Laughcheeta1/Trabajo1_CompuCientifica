import json

notebook_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "numerical_cols =" in source:
            print(f"--- Cell {i} Source ---")
            for j, line in enumerate(cell["source"]):
                print(f"Line {j}: {repr(line)}")
