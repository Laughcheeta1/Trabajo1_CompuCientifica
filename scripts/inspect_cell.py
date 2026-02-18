import json

notebook_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_lines = cell["source"]
        source_text = "".join(source_lines)
        if "numerical_cols =" in source_text:
            print(f"--- Cell {i} Content ---")
            print(json.dumps(source_lines, indent=2))
