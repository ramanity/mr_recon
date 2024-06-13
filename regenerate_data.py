import os
import shutil
import random

hr_train_dir = '/home/jupyter/mri_recon/data/hr_train'
hr_val_dir = '/home/jupyter/mri_recon/data/hr_val'
base_target_dir = '/home/jupyter/mri_recon/data/'

train_dir = os.path.join(base_target_dir, 'train')
val_dir = os.path.join(base_target_dir, 'val')
test_dir = os.path.join(base_target_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_files = [os.path.join(hr_train_dir, f) for f in os.listdir(hr_train_dir) if os.path.isfile(os.path.join(hr_train_dir, f))]
val_files = [os.path.join(hr_val_dir, f) for f in os.listdir(hr_val_dir) if os.path.isfile(os.path.join(hr_val_dir, f))]
all_files = train_files + val_files
random.shuffle(all_files)
selected_files = all_files[:1000]
train_split = int(0.8 * len(selected_files))
val_split = int(0.1 * len(selected_files))
test_split = len(selected_files) - train_split - val_split

train_files = selected_files[:train_split]
val_files = selected_files[train_split:train_split + val_split]
test_files = selected_files[train_split + val_split:]

def move_files(files, target_dir):
    for file in files:
        shutil.copy(file, target_dir)

move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print(f"Train set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")
