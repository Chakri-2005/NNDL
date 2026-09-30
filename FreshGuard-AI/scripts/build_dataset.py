import os
import shutil
import random

def build_dataset(raw_dir, target_dir):
    random.seed(42)
    
    # 1. Classes to keep and rename
    class_map = {
        'Apple_Good': 'Apple_Fresh',
        'Apple_Bad': 'Apple_Rotten',
        'Banana_Good': 'Banana_Fresh',
        'Banana_Bad': 'Banana_Rotten',
        'Orange_Good': 'Orange_Fresh',
        'Orange_Bad': 'Orange_Rotten'
    }
    
    splits = ['train', 'val', 'test']
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    # Create target directories
    for split in splits:
        for new_class in class_map.values():
            os.makedirs(os.path.join(target_dir, split, new_class), exist_ok=True)
            
    # Track stats
    stats = {new_class: {'train': 0, 'val': 0, 'test': 0} for new_class in class_map.values()}
    
    print("Copying and splitting files...")
    for old_class, new_class in class_map.items():
        src_path = os.path.join(raw_dir, old_class)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found")
            continue
            
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort() # Ensure deterministic order before shuffling
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Copy Train
        for img in train_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(target_dir, 'train', new_class, img))
            stats[new_class]['train'] += 1
            
        # Copy Val
        for img in val_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(target_dir, 'val', new_class, img))
            stats[new_class]['val'] += 1
            
        # Copy Test
        for img in test_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(target_dir, 'test', new_class, img))
            stats[new_class]['test'] += 1
            
    # Generate /results/dataset_split.md
    with open("results/dataset_split.md", "w") as f:
        f.write("# Dataset Split Summary\n\n")
        
        # Paragraph as requested in Prompt 2, step 5
        f.write("### Scope Decision Justification\n")
        f.write("For this initial deployment, we have scoped the dataset to the three most commercially dominant fruits (Apple, Banana, Orange). This ensures our flagship Attention-CNN model can focus on learning highly robust, generalized freshness features without being penalized by the high intra-class variance of the full 12-class FruitsGB dataset. Furthermore, relabeling the 'Good/Bad' dataset nomenclature to 'Fresh/Rotten' specifically aligns the model's vocabulary with our problem statement—focusing on food safety and spoilage detection rather than generic cosmetic defects.\n\n")
        
        f.write("### Image Counts by Split\n\n")
        f.write("| Class | Train (70%) | Val (15%) | Test (15%) | Total |\n")
        f.write("|---|---|---|---|---|\n")
        
        total_train = total_val = total_test = 0
        for cls, s in stats.items():
            tot = s['train'] + s['val'] + s['test']
            f.write(f"| {cls} | {s['train']} | {s['val']} | {s['test']} | {tot} |\n")
            total_train += s['train']
            total_val += s['val']
            total_test += s['test']
            
        grand_total = total_train + total_val + total_test
        f.write(f"| **TOTAL** | **{total_train}** | **{total_val}** | **{total_test}** | **{grand_total}** |\n")

    print("Phase 1 complete. Dataset built and split written to results/dataset_split.md")

if __name__ == "__main__":
    raw_dir = "Top Indian Fruits Dataset_Clean Images/FruitDataSet of 256x256 Size_12000 Images"
    target_dir = "data"
    build_dataset(raw_dir, target_dir)
