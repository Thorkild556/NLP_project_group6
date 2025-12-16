import json
from pathlib import Path

def fix_widgets_metadata(notebook_path, output_path=None):
    notebook_path = Path(notebook_path)
    if output_path is None:
        output_path = notebook_path.with_name(notebook_path.stem + "_fixed.ipynb")
    else:
        output_path = Path(output_path)

    # Load notebook
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Iterate over all cells and fix metadata.widgets
    for cell in nb.get("cells", []):
        widgets = cell.get("metadata", {}).get("widgets")
        if widgets is not None:
            for widget_id, widget_data in widgets.items():
                if "state" not in widget_data:
                    widget_data["state"] = {}  # add empty state
            # OR, if you prefer, remove widgets completely:
            # cell["metadata"].pop("widgets", None)

    # Save fixed notebook
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Fixed notebook saved to: {output_path}")

# Example usage
fix_widgets_metadata("CalculateLossThreshold.ipynb")
