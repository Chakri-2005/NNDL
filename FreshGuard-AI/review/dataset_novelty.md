# FreshGuard AI — Dataset Scope & Deduplication Methodology

## 1. Scope & Categorization
The project focuses on three commercially critical fruits: **Apple**, **Banana**, and **Orange**, split into **Fresh** and **Rotten** quality states across 6 standardized target classes:
1. `Apple_Fresh`
2. `Apple_Rotten`
3. `Banana_Fresh`
4. `Banana_Rotten`
5. `Orange_Fresh`
6. `Orange_Rotten`

## 2. Dataset Novelty & Framing Justification
Unlike traditional visual fruit classification datasets (e.g., Fruits-360) that focus solely on species identification under isolated lab conditions, FreshGuard AI specifically addresses **quality and spoilage grading**. Relabeling raw dataset terminology ("Good/Bad") to "Fresh/Rotten" aligns the model's vocabulary directly with consumer food safety and agricultural waste reduction.

## 3. The Data Leakage Problem in Prior Work
In image sequence capture setups, multiple burst photos are taken of the exact same physical fruit item from slight variations in angle or illumination. Standard random image-level splitting (`random_split()`) scatters burst photos of the same physical fruit across both training and validation sets. Models memorize background context or surface markings, reporting artificially inflated accuracies (e.g. 98–100%) that fail on real-world unseen fruits.

## 4. Leakage-Safe Deduplication & Grouping Methodology
FreshGuard AI solves data leakage via a two-stage topological pipeline:
1. **Exact SHA-256 Hashing**: Identifies and flags byte-identical duplicate files.
2. **Perceptual Difference Hashing (dHash)**: Computes 64-bit gradient hashes. Pairs of images with Hamming distance $D \le 8$ are linked as near-duplicate burst photos of the same fruit item.
3. **Union-Find Graph Clustering**: Connected component analysis clusters linked images into discrete **physical fruit group IDs**.
4. **Group-Stratified Partitioning**: Partitioning (80% Train, 10% Val, 10% Test) is performed at the group level, guaranteeing that all photos of a given fruit item remain strictly isolated within a single split.

## 5. Partition Manifest & Verification
- **Total Images**: 900 images.
- **Unique Physical Fruit Groups**: 78 isolated groups.
- **Train Set**: 624 images (81.6% groups).
- **Validation Set**: 224 images (9.2% groups).
- **Test Set**: 52 images (9.2% groups).
- **Group Leakage**: **0 groups** crossing train/val/test boundaries.
