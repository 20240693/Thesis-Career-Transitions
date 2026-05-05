import json
from pathlib import Path

for path in Path(".").glob("*.ipynb"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "widgets" in nb.get("metadata", {}):
        nb["metadata"].pop("widgets", None)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"Fixed: {path.name}")
    else:
        print(f"OK: {path.name}")