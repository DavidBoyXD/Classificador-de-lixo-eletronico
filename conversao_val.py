import os
import random
import shutil
from glob import glob

# Caminhos base
base_dir = "dataset_processado/MobilePhone"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Cria a estrutura de pastas YOLO
for folder in ["train", "val"]:
    os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, folder), exist_ok=True)

# Lista todas as imagens
image_files = glob(os.path.join(base_dir, "*.jpg"))

# Embaralha e separa em treino/validação (80/20)
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_pairs(files, split):
    for img_path in files:
        name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(label_path):
            continue
        shutil.move(img_path, os.path.join(images_dir, split, name))
        shutil.move(label_path, os.path.join(labels_dir, split, os.path.basename(label_path)))

move_pairs(train_files, "train")
move_pairs(val_files, "val")

print("✅ Dataset reorganizado no formato YOLOv8!")
