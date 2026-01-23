import numpy as np
import json
import sys
from pathlib import Path

# Usage: python npz_to_json.py path/to/input.npz path/to/output.json

def npz_to_json(npz_path, json_path):
    data = np.load(npz_path, allow_pickle=True)
    arr = data[data.files[0]]
    # arr is an array of dicts: {'file': ..., 'label': ...}
    samples = [dict(sample) for sample in arr]
    with open(json_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python npz_to_json.py path/to/input.npz path/to/output.json")
        sys.exit(1)
    npz_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    npz_to_json(npz_path, json_path)
