import os
import numpy as np
from pathlib import Path

# Always make sure this is a Path
DATASET_ROOT = Path(__file__).resolve().parent / "Briareo_rgb"

SPLITS = ['train', 'val', 'test']

SPLIT_FOLDERS = {
    'train': 'train',
    'val': 'val',
    'test': 'test',
}

def get_label_from_folder(folder_name):
    if folder_name.startswith('g') and folder_name[1:].isdigit():
        return int(folder_name[1:])
    return None

def collect_samples(split_folder: Path):
    samples = []

    for subject in sorted(os.listdir(split_folder)):
        subject_path = split_folder / subject
        if not subject_path.is_dir():
            continue

        for gesture in sorted(os.listdir(subject_path)):
            gesture_path = subject_path / gesture
            if not gesture_path.is_dir():
                continue

            label = get_label_from_folder(gesture)
            if label is None:
                continue

            for sequence in sorted(os.listdir(gesture_path)):
                seq_path = gesture_path / sequence
                rgb_dir = seq_path / "rgb"

                if not rgb_dir.exists():
                    continue

                frame_files = sorted(
                    str(rgb_dir / f)
                    for f in os.listdir(rgb_dir)
                    if f.endswith(".png")
                )

                if frame_files:
                    samples.append({
                        "data": frame_files,
                        "label": label
                    })

    return samples

def main():
    for split in SPLITS:
        split_folder = DATASET_ROOT / SPLIT_FOLDERS[split]

        if not split_folder.exists():
            print(f"⚠️  Split folder not found: {split_folder}")
            continue

        samples = collect_samples(split_folder)

        out_dir = DATASET_ROOT / "splits" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"rgb_{split}.npz"
        np.savez_compressed(out_path, np.array(samples, dtype=object))

        print(f"✅ Saved {len(samples)} samples → {out_path}")

if __name__ == "__main__":
    main()
