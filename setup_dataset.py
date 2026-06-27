"""
setup_dataset.py
================
Automatically organizes your Kaggle images into the correct
train/val folder structure needed for training.

Usage:
    python setup_dataset.py --source "E:\path\to\colored_images" --dest "data\eyepacs"

Your source folder should have:
    colored_images/
        No_DR/
        Mild/
        Moderate/
        Severe/
        Proliferate_DR/
"""

import os, shutil, argparse, random
from pathlib import Path
from tqdm import tqdm

FOLDER_TO_CLASS = {
    "No_DR":         0,
    "Mild":          1,
    "Moderate":      2,
    "Severe":        3,
    "Proliferate_DR": 4,
}

def setup_dataset(source_dir, dest_dir, val_split=0.2, max_per_class=None):
    source = Path(source_dir)
    dest   = Path(dest_dir)

    total_copied = 0

    for folder_name, class_idx in FOLDER_TO_CLASS.items():
        src_folder = source / folder_name
        if not src_folder.exists():
            print(f"WARNING: Folder not found: {src_folder}")
            continue

        images = list(src_folder.glob("*.png")) + \
                 list(src_folder.glob("*.jpg")) + \
                 list(src_folder.glob("*.jpeg"))

        if not images:
            print(f"WARNING: No images found in {src_folder}")
            continue

        random.shuffle(images)

        if max_per_class:
            images = images[:max_per_class]

        split      = int(len(images) * (1 - val_split))
        train_imgs = images[:split]
        val_imgs   = images[split:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = dest / split_name / str(class_idx)
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_imgs, desc=f"{folder_name} → {split_name}/{class_idx}"):
                shutil.copy2(img_path, out_dir / img_path.name)
                total_copied += 1

        print(f"  {folder_name}: {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"\nDone! {total_copied} images organized in: {dest}")
    print("\nFolder structure created:")
    print(f"  {dest}/train/0  (No DR)")
    print(f"  {dest}/train/1  (Mild)")
    print(f"  {dest}/train/2  (Moderate)")
    print(f"  {dest}/train/3  (Severe)")
    print(f"  {dest}/train/4  (Proliferative)")
    print(f"  {dest}/val/0-4  (validation sets)")
    print(f"\nNow run training:")
    print(f"  python train_professional.py --data_dir {dest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to colored_images folder")
    parser.add_argument("--dest",   default="data/eyepacs", help="Output path")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max images per class (optional)")
    args = parser.parse_args()

    random.seed(42)
    setup_dataset(args.source, args.dest, args.val_split, args.max_per_class)
