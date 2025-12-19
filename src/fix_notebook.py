import json
from pathlib import Path

# we were getting rendering errors when uploading the notebook to the GitHub (the one we downloaded from colab)
# please run this script to avoid such error

def fix_colab_widgets(ipynb_path, output_path=None):
    ipynb_path = Path(ipynb_path)

    if output_path is None:
        output_path = ipynb_path.with_name(ipynb_path.stem + "_fixed.ipynb")
    else:
        output_path = Path(output_path)

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # 🔥 THIS is the actual problem location
    if "metadata" in nb and "widgets" in nb["metadata"]:
        print("Removing broken metadata.widgets")
        nb["metadata"].pop("widgets")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Fixed notebook saved as: {output_path}")

# Example usage
fix_colab_widgets("../lab/fine_tune_models.ipynb")
