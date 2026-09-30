"""
FreshGuard AI — Leakage-Safe Dataset Preparation Script
Performs:
1. Generates synthetic raw dataset images if external FruitsGB folder is not on disk.
2. Exact SHA-256 duplicate detection.
3. Perceptual Difference Hashing (dHash) near-duplicate detection.
4. Group-based isolation (Union-Find) to ensure near-duplicate photos of the same physical fruit stay in ONE split.
5. Stratified group splitting into 80% Train, 10% Validation, 10% Test.
6. Saves clean split CSVs and physical files in data/.
"""
import os
import sys
import hashlib
import json
import shutil
import random
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_synthetic_raw_dataset():
    """Generates realistic structured fruit images for the 6 target classes if raw dataset is absent."""
    raw_dir = config.RAW_DATASET_DIR
    print(f"Raw dataset folder absent at '{raw_dir}'. Generating reproducible 6-class dataset...")
    os.makedirs(raw_dir, exist_ok=True)

    fruit_colors = {
        "Apple_Good": (220, 38, 38),     # Bright Red
        "Apple_Bad": (120, 50, 40),      # Dark Brownish Red
        "Banana_Good": (234, 179, 8),    # Bright Yellow
        "Banana_Bad": (100, 80, 20),     # Dark Brown
        "Orange_Good": (249, 115, 22),   # Bright Orange
        "Orange_Bad": (150, 80, 20)      # Dull Brownish Orange
    }

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    images_per_class = 150  # 150 images * 6 classes = 900 total images
    burst_size = 3          # 3 images per physical fruit group

    for raw_cls, base_rgb in fruit_colors.items():
        cls_dir = os.path.join(raw_dir, raw_cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        num_fruit_items = images_per_class // burst_size

        for item_idx in range(num_fruit_items):
            center_x = random.randint(110, 146)
            center_y = random.randint(110, 146)
            radius = random.randint(60, 90)

            for burst_idx in range(burst_size):
                img = Image.new('RGB', (256, 256), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)

                var_r = min(255, max(0, base_rgb[0] + random.randint(-15, 15)))
                var_g = min(255, max(0, base_rgb[1] + random.randint(-15, 15)))
                var_b = min(255, max(0, base_rgb[2] + random.randint(-15, 15)))
                color = (var_r, var_g, var_b)

                bbox = [
                    center_x - radius + random.randint(-2, 2),
                    center_y - radius + random.randint(-2, 2),
                    center_x + radius + random.randint(-2, 2),
                    center_y + radius + random.randint(-2, 2)
                ]
                draw.ellipse(bbox, fill=color, outline=(40, 40, 40), width=2)

                if 'Bad' in raw_cls:
                    for _ in range(random.randint(3, 7)):
                        spot_x = center_x + random.randint(-radius//2, radius//2)
                        spot_y = center_y + random.randint(-radius//2, radius//2)
                        spot_r = random.randint(8, 20)
                        draw.ellipse([spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r], fill=(60, 40, 20))

                img_path = os.path.join(cls_dir, f"item_{item_idx:03d}_frame_{burst_idx:02d}.jpg")
                img.save(img_path, quality=95)

    print(f"Generated synthetic raw dataset in {raw_dir}.\n")

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image_path, hash_size=8):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getchannel(0).getdata()) if hasattr(img, 'getchannel') else list(img.getdata())
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(pixel_left > pixel_right)
            decimal_val = 0
            for bit in difference:
                decimal_val = (decimal_val << 1) | bit
            return decimal_val
    except Exception as e:
        print(f"Warning: Could not compute hash for {image_path}: {e}")
        return 0

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def prepare_leakage_safe_dataset():
    random.seed(config.SEED)
    print("="*60)
    print("  FreshGuard AI -- Leakage-Safe Dataset Preparation")
    print("="*60)
    
    raw_dir = config.RAW_DATASET_DIR
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        generate_synthetic_raw_dataset()

    # 1. Collect all images for the 6 target classes
    image_records = []
    print("\n[Step 1/5] Indexing raw images...")
    for raw_cls, target_cls in config.RAW_CLASS_MAPPING.items():
        cls_path = os.path.join(raw_dir, raw_cls)
        if not os.path.exists(cls_path):
            continue
            
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            image_records.append({
                'raw_filepath': fpath,
                'filename': fname,
                'raw_class': raw_cls,
                'target_class': target_cls,
                'label': config.CLASS_TO_IDX[target_cls]
            })
            
    df = pd.DataFrame(image_records)
    print(f"Indexed {len(df)} total images across 6 target classes.")
    
    # 2. Exact SHA-256 Hash Computation
    print("\n[Step 2/5] Computing SHA-256 hashes for exact duplicate detection...")
    sha_list = []
    for fpath in df['raw_filepath']:
        sha_list.append(compute_sha256(fpath))
    df['sha256'] = sha_list
    
    exact_duplicates_count = df.duplicated(subset=['sha256']).sum()
    print(f"Found {exact_duplicates_count} exact byte-level duplicate images.")
    
    # 3. Perceptual dHash Computation & Near-Duplicate Grouping
    print("\n[Step 3/5] Computing dHash perceptual hashes for near-duplicate grouping...")
    dhash_list = []
    for fpath in df['raw_filepath']:
        dhash_list.append(compute_dhash(fpath))
    df['dhash'] = dhash_list
    
    uf = UnionFind(range(len(df)))
    
    sha_groups = defaultdict(list)
    for idx, sha in enumerate(df['sha256']):
        sha_groups[sha].append(idx)
    for indices in sha_groups.values():
        for i in range(1, len(indices)):
            uf.union(indices[0], indices[i])
            
    class_indices = defaultdict(list)
    for idx, target_cls in enumerate(df['target_class']):
        class_indices[target_cls].append(idx)
        
    near_dup_edges = 0
    for target_cls, indices in class_indices.items():
        for i in range(len(indices)):
            idx_i = indices[i]
            hash_i = df.at[idx_i, 'dhash']
            if hash_i == 0:
                continue
            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                hash_j = df.at[idx_j, 'dhash']
                if hash_j == 0:
                    continue
                if hamming_distance(hash_i, hash_j) <= 8:
                    uf.union(idx_i, idx_j)
                    near_dup_edges += 1
                    
    print(f"Connected {near_dup_edges} near-duplicate image pairs into group clusters.")
    
    group_ids = [uf.find(i) for i in range(len(df))]
    df['group_id'] = group_ids
    num_unique_groups = df['group_id'].nunique()
    print(f"Total unique physical fruit image groups identified: {num_unique_groups} (from {len(df)} images).")
    
    # 4. Stratified Group Split (80% Train, 10% Val, 10% Test)
    print("\n[Step 4/5] Performing group-stratified split (80% Train, 10% Val, 10% Test)...")
    train_records, val_records, test_records = [], [], []
    
    for target_cls in config.CLASS_NAMES:
        cls_df = df[df['target_class'] == target_cls]
        group_df = cls_df.groupby('group_id').first().reset_index()
        groups = group_df['group_id'].tolist()
        random.shuffle(groups)
        
        n_groups = len(groups)
        n_train = max(1, int(0.80 * n_groups))
        n_val = max(1, int(0.10 * n_groups))
        
        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train:n_train + n_val])
        test_groups = set(groups[n_train + n_val:])
        if not test_groups and len(val_groups) > 1:
            val_list = list(val_groups)
            test_groups = {val_list.pop()}
            val_groups = set(val_list)
        
        cls_train = cls_df[cls_df['group_id'].isin(train_groups)]
        cls_val = cls_df[cls_df['group_id'].isin(val_groups)]
        cls_test = cls_df[cls_df['group_id'].isin(test_groups)]
        
        train_records.append(cls_train)
        val_records.append(cls_val)
        test_records.append(cls_test)
        
    df_train = pd.concat(train_records).reset_index(drop=True)
    df_val = pd.concat(val_records).reset_index(drop=True)
    df_test = pd.concat(test_records).reset_index(drop=True)
    
    print(f"Split completed: Train = {len(df_train)} images, Val = {len(df_val)} images, Test = {len(df_test)} images.")
    
    # Verify ZERO group leakage
    train_groups_set = set(df_train['group_id'])
    val_groups_set = set(df_val['group_id'])
    test_groups_set = set(df_test['group_id'])
    
    leakage_tv = train_groups_set.intersection(val_groups_set)
    leakage_tt = train_groups_set.intersection(test_groups_set)
    leakage_vt = val_groups_set.intersection(test_groups_set)
    
    total_leakage = len(leakage_tv) + len(leakage_tt) + len(leakage_vt)
    if total_leakage == 0:
        print("[OK] VERIFIED: Absolute 0 group leakage across Train/Val/Test splits!")
    else:
        print(f"[WARNING] Found {total_leakage} leaking groups across splits!")
        
    # 5. Copy files to data/ directory and save CSV manifests
    print("\n[Step 5/5] Structuring physical files and saving split CSVs...")
    
    splits_dict = {'train': df_train, 'val': df_val, 'test': df_test}
    
    for split_name, split_df in splits_dict.items():
        processed_paths = []
        for idx, row in split_df.iterrows():
            target_dir = os.path.join(config.DATA_DIR, split_name, row['target_class'])
            os.makedirs(target_dir, exist_ok=True)
            dst_path = os.path.join(target_dir, row['filename'])
            shutil.copy(row['raw_filepath'], dst_path)
            rel_path = os.path.relpath(dst_path, config.BASE_DIR)
            processed_paths.append(rel_path)
            
        split_df['filepath'] = processed_paths
        csv_path = os.path.join(config.SPLITS_DIR, f"{split_name}.csv")
        save_cols = ['filepath', 'target_class', 'label', 'group_id', 'sha256']
        split_df[save_cols].to_csv(csv_path, index=False)
        print(f" Saved split manifest: {csv_path} ({len(split_df)} rows)")
        
    report = {
        'total_images': len(df),
        'exact_sha256_duplicates': int(exact_duplicates_count),
        'unique_physical_fruit_groups': num_unique_groups,
        'train_count': len(df_train),
        'val_count': len(df_val),
        'test_count': len(df_test),
        'group_leakage_count': total_leakage
    }
    report_path = os.path.join(config.SPLITS_DIR, "deduplication_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nSaved deduplication report to {report_path}")
    print("Dataset preparation complete!\n")

if __name__ == "__main__":
    prepare_leakage_safe_dataset()
