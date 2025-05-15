import os
import pandas as pd
import shutil
import random

# Define paths
src_dir = './temporary/ETL4/ETL4C_unpack'
train_dir = './data/train'
val_dir = './data/val'

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read metadata
meta_df = pd.read_csv(src_dir + '/meta.csv')

# Get unique characters
unique_chars = meta_df['char'].unique()
print(f"Found {len(unique_chars)} unique characters in metadata")

# Dictionary to store image paths for each character
char_images = {}

# List all files in the source directory to debug
all_files = os.listdir(src_dir)
print(f"Found {len(all_files)} files in the source directory")
png_files = [f for f in all_files if f.endswith('.png')]
print(f"Found {len(png_files)} PNG files in the source directory")

# Group image paths by character
for _, row in meta_df.iterrows():
    char = row['char']
    img_idx = row.name
    img_path = os.path.join(src_dir, f'{img_idx:05d}.png')
    
    # Skip if the image doesn't exist
    if not os.path.exists(img_path):
        # Try the alternative format seen in the directory
        alt_img_path = os.path.join(src_dir, f'{img_idx + 6000:05d}.png')
        if not os.path.exists(alt_img_path):
            continue
        img_path = alt_img_path
    
    if char not in char_images:
        char_images[char] = []
    
    char_images[char].append(img_path)

# Print debug info
found_chars = len(char_images)
found_images = sum(len(imgs) for imgs in char_images.values())
print(f"Found images for {found_chars} characters")
print(f"Found {found_images} total images")

# Manual train/val split without sklearn
random.seed(42)  # 同じ結果を得るための固定シード

# Split data into train and validation sets
for char, img_paths in char_images.items():
    # Create character directories
    char_train_dir = os.path.join(train_dir, char)
    char_val_dir = os.path.join(val_dir, char)
    
    os.makedirs(char_train_dir, exist_ok=True)
    os.makedirs(char_val_dir, exist_ok=True)
    
    # Manual split into train and validation (80/20)
    if len(img_paths) > 1:
        # シャッフルしてからリストを分割
        shuffled_paths = img_paths.copy()
        random.shuffle(shuffled_paths)
        
        # 80%をトレーニングに、20%を検証に
        val_size = int(len(shuffled_paths) * 0.2)
        train_paths = shuffled_paths[val_size:]
        val_paths = shuffled_paths[:val_size]
    else:
        # 1つしか画像がない場合はトレーニングに入れる
        train_paths, val_paths = img_paths, []
    
    # Copy images to respective directories
    for i, path in enumerate(train_paths):
        dst_path = os.path.join(char_train_dir, f'{i:04d}.png')
        shutil.copy(path, dst_path)
    
    for i, path in enumerate(val_paths):
        dst_path = os.path.join(char_val_dir, f'{i:04d}.png')
        shutil.copy(path, dst_path)

print(f"Dataset prepared with {len(unique_chars)} hiragana characters")
print(f"- Training images: {sum(len(files) for _, _, files in os.walk(train_dir))}")
print(f"- Validation images: {sum(len(files) for _, _, files in os.walk(val_dir))}") 