import os
import shutil

# IDs das classes no YOLO
SMARTPHONE_ID = 0  # Altere para o ID do smartphone no seu YAML

# Caminho da base original (onde está seu dataset YOLO)
BASE_DIR = "dataset/mobilephone"
# Caminho de destino
OUTPUT_DIR = "dataset_binary"

# Conjuntos (train, valid, test)
SETS = ["train", "valid", "test"]

for subset in SETS:
    print(f"\nProcessando {subset}...")
    image_dir = os.path.join(BASE_DIR, subset, "images")
    label_dir = os.path.join(BASE_DIR, subset, "labels")

    out_smartphone = os.path.join(OUTPUT_DIR, subset, "smartphone")
    out_other = os.path.join(OUTPUT_DIR, subset, "other")
    os.makedirs(out_smartphone, exist_ok=True)
    os.makedirs(out_other, exist_ok=True)

    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            continue  # pula se a imagem não existir

        # Lê o arquivo de label
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Se alguma linha tiver o ID do smartphone
        if any(line.startswith(str(SMARTPHONE_ID)) for line in lines):
            dest = out_smartphone
        else:
            dest = out_other

        shutil.copy(image_path, os.path.join(dest, image_name))

print("\n✅ Conversão concluída!")
