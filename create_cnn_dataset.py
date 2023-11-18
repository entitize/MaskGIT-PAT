import numpy as np
import os
from sklearn.model_selection import train_test_split
import json

# # Prepare Dataset
def crop_and_norm(image, original_width, original_height, target_size, global_min, global_max):
    '''
    This function was used to create the dataset in '/groups/mlprojects/pat/pat_cnn'.
    '''
    # Crop the image to the target size
    left = original_width // 2 - target_size[0] // 2
    top = original_height // 2 - target_size[1] // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    cropped_image = image[top:bottom, left:right]

    # Reshape to be expected size
    cropped_image_array = np.array(cropped_image).reshape(*target_size, 1)
    # normalize to be from 0 to 1
    normalized_image = (cropped_image_array - global_min) / (global_max - global_min)

    return normalized_image

original_width, original_height = 100, 188
target_size = (64, 64)

new_dir = '/groups/mlprojects/pat/pat_norm_crop'
os.makedirs(new_dir, exist_ok=True)
os.makedirs(os.path.join(new_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'test'), exist_ok=True)

data_dir = "/groups/mlprojects/pat/pat_np/original"
data = [] # image arrays
labels = [] # file names
print("Loading data...")
for f in os.listdir(data_dir):
    if f.endswith('.npy'):
        data.append(np.load(os.path.join(data_dir, f)))
        labels.append(f)

print(len(data))
# Find the global minimum and maximum
combined_array = np.concatenate(data, axis=0)
global_min = np.min(combined_array)
global_max = np.max(combined_array)

# save global min/max and cropping locations to config file
config = {'min':global_min, 'max' : global_max}
config['left'] = original_width // 2 - target_size[0] // 2
config['top'] = original_height // 2 - target_size[1] // 2
config['size'] = target_size
with open(os.path.join(new_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f)

print("Cropping and normalizing...")
data = [crop_and_norm(d, original_width, original_height, target_size, global_min, global_max) for d in data]

# 80/10/10 train/val/test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print("Saving data...")
for i, arr in enumerate(X_train):
    np.save(os.path.join(new_dir, 'train', y_train[i]), arr)

for i, arr in enumerate(X_val):
    np.save(os.path.join(new_dir, 'val', y_val[i]), arr)

for i, arr in enumerate(X_test):
    np.save(os.path.join(new_dir, 'test', y_test[i]), arr)

print("Done")
