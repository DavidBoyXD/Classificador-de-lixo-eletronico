import os
import random
import shutil
from glob import glob
import yaml

# === CONFIGURAÇÕES ===
BASE_DIR = "dataset_processado/MobilePhone"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLIT_RATIOS = {"train": 0.8, "val": 0.15, "test": 0.05}
CLASS_NAMES = ["smartphone"]  # Nome da classe

# === CRIAR ESTRUTURA ===
for folder_type in ["images", "labels"]:
    for split in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(BASE_DIR, folder_type, split), exist_ok=True)

# === LISTAR TODAS AS IMAGENS ===
image_files = [f for f in glob(os.path.join(BASE_DIR, "*")) if f.lower().endswith(IMG_EXTENSIONS)]
random.shuffle(image_files)

# === DIVIDIR ===
n_total = len(image_files)
train_end = int(SPLIT_RATIOS["train"] * n_total)
val_end = train_end + int(SPLIT_RATIOS["val"] * n_total)

splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# === MOVER IMAGENS E LABELS ===
for split, files in splits.items():
    for img_path in files:
        img_name = os.path.basename(img_path)
        label_path = os.path.splitext(img_path)[0] + ".txt"

        # Destinos
        dest_img = os.path.join(BASE_DIR, "images", split, img_name)
        dest_label = os.path.join(BASE_DIR, "labels", split, os.path.basename(label_path))

        # Verifica existência e move
        if os.path.exists(label_path):
            shutil.move(img_path, dest_img)
            shutil.move(label_path, dest_label)
        else:
            print(f"⚠️ Label não encontrado para: {img_name}")

print("\n✅ Dataset reorganizado com sucesso!\n")

# === CRIAR data.yaml ===
yaml_path = os.path.join(BASE_DIR, "data.yaml")
data_config = {
    "train": f"{BASE_DIR}/images/train",
    "val": f"{BASE_DIR}/images/val",
    "test": f"{BASE_DIR}/images/test",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES
}

with open(yaml_path, "w") as f:
    yaml.dump(data_config, f)

print(f"✅ Arquivo 'data.yaml' criado em: {yaml_path}")
print("\n📁 Estrutura final esperada:")
print("""
dataset_processado/MobilePhone/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
""")
