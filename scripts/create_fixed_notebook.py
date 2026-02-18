import json
import re
import os

input_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1.ipynb"
output_path = r"c:\Users\57321\OneDrive - Universidad EIA\9noSemestre\compCientifica\entrega1\notebooks\entrega1_2.ipynb"

try:
    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Join source to handle multi-line string concatenation
            source_str = "".join(cell["source"])

            # Pattern: 'NumStorePurchases' followed by optional whitespace/newlines, then 'NumWebVisitsMonth'
            # We want to insert a comma if it's missing.
            # We explicitly look for NO comma between them.
            pattern = r"('NumStorePurchases')(\s*)('NumWebVisitsMonth')"

            if re.search(pattern, source_str):
                print("Found missing comma pattern in a cell.")
                # Replace with comma
                new_source_str = re.sub(pattern, r"\1,\2\3", source_str)

                # Update cell source
                # splitlines(keepends=True) keeps the \n at end of lines
                cell["source"] = new_source_str.splitlines(keepends=True)
                fixed = True

    # Write the new notebook
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    if fixed:
        print(f"Successfully created fixed notebook: {output_path}")
    else:
        print(f"Pattern not found, but created copy: {output_path}")

except Exception as e:
    print(f"Error: {e}")
