import os
import random
import shutil

# Set paths
frames_dir = "C:/Users/sinzs/Downloads/Celeb-DF/frames"
output_dir = "C:/Users/sinzs/Downloads/Celeb-DF/split_frames"
os.makedirs(output_dir, exist_ok=True)

def split_data(files, split_ratio):
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)
    return files[:split_idx], files[split_idx:]

def copy_files(files, label, split):
    dest_dir = os.path.join(output_dir, split, label)
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        src = os.path.join(frames_dir, label, file)
        dst = os.path.join(dest_dir, file)
        if os.path.isfile(src):  # Only copy actual files
            shutil.copy(src, dst)

# Gather files from respective folders
real_files = os.listdir(os.path.join(frames_dir, "real"))
fake_files = os.listdir(os.path.join(frames_dir, "fake"))

# Split into train (70%), val (15%), test (15%)
real_train, real_temp = split_data(real_files, 0.7)
real_val, real_test = split_data(real_temp, 0.5)

fake_train, fake_temp = split_data(fake_files, 0.7)
fake_val, fake_test = split_data(fake_temp, 0.5)

# Copy files to appropriate directories
copy_files(real_train, 'real', 'train')
copy_files(real_val, 'real', 'val')
copy_files(real_test, 'real', 'test')

copy_files(fake_train, 'fake', 'train')
copy_files(fake_val, 'fake', 'val')
copy_files(fake_test, 'fake', 'test')

print("✅ Frames organized into train/val/test splits with labels.")
