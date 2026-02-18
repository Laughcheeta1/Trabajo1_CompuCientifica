import json
import re

notebook_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Iterate over cells
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])

        # Check for the pattern with flexible quotes
        # Pattern: Quote, String1, Quote, Spaces/Newlines, Quote, String2, Quote
        pattern = r"(['\"]NumStorePurchases['\"])(\s*)(['\"]NumWebVisitsMonth['\"])"

        if re.search(pattern, source_text):
            print("Found pattern in cell source.")
            new_source_text = re.sub(pattern, r"\1,\2\3", source_text)

            lines = new_source_text.splitlines(keepends=True)
            cell["source"] = lines

            found = True
            print("Fixed missing comma.")

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4)
    print("Notebook saved.")
else:
    print("Could not find the target string to fix.")
