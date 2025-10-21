# -*- coding: utf-8 -*-

import os
import cv2
from tqdm import tqdm
import yaml

# --- Configurações principais ---
# Classes alvo para o modelo binário
TARGET_CLASSES = ["Desktop-PC", "Smartphone"]
CLASS_ID_MAPPING = {name: i for i, name in enumerate(TARGET_CLASSES)}  # Desktop-PC=0, Smartphone=1

# Datasets originais (substitua pelos caminhos corretos)
RAW_DATA_DIRS = ["dataset/e-waste", "dataset/smartphonedetection", "dataset/mobilephone", "dataset/phonefinder"]

# Pasta de saída unificada
PROCESSED_DATA_DIR = "dataset/processed_2_classes"

# Subsets
SUBSETS = ["train", "valid", "test"]

# --- Função de pré-processamento ---
def apply_preprocessing_pipeline(image):
    """Converte para grayscale, redimensiona, aplica Gaussian blur e equaliza histograma."""
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    resized_image = cv2.resize(gray_image, (224, 224))
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image

# --- Função de processamento ---
def process_dataset(base_dir, output_dir):
    """Processa todos os subsets de um dataset e salva na pasta unificada."""

    # Tenta ler o YAML se existir (para pegar os nomes das classes do dataset)
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, "r") as f:
                data_config = yaml.safe_load(f)
            original_class_names = data_config.get("names", [])
        except Exception as e:
            print(f"Aviso: não foi possível ler {data_yaml_path}: {e}")
            original_class_names = []
    else:
        original_class_names = []

    # Se não tiver YAML ou names, usa uma lista vazia (somente filtragem por TARGET_CLASSES)
    if not original_class_names:
        print(f"Aviso: YAML não encontrado ou inválido em {base_dir}, usando nomes padrões do TARGET_CLASSES")
        original_class_names = TARGET_CLASSES.copy()

    for split in SUBSETS:
        img_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        if not os.path.exists(img_dir):
            print(f"Aviso: {img_dir} não encontrado, pulando...")
            continue

        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for filename in tqdm(image_files, desc=f"Processando {split} em {os.path.basename(base_dir)}"):
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(label_path):
                print(f"Aviso: label não encontrado para {filename}, pulando...")
                continue

            try:
                with open(label_path, "r") as f:
                    class_id_raw = int(f.readline().strip().split()[0])

                # Pega o nome original da classe a partir do YAML ou usa ID se YAML ausente
                if class_id_raw < len(original_class_names):
                    class_name_raw = original_class_names[class_id_raw]
                else:
                    class_name_raw = f"Class_{class_id_raw}"

                # Filtra apenas TARGET_CLASSES
                if class_name_raw not in TARGET_CLASSES:
                    continue

                # Cria pasta de saída correta
                class_output_dir = os.path.join(output_dir, split, class_name_raw)
                os.makedirs(class_output_dir, exist_ok=True)

                # Carrega e processa imagem
                img_path = os.path.join(img_dir, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Aviso: imagem não pôde ser carregada {filename}")
                    continue

                processed_image = apply_preprocessing_pipeline(image)

                # Salva a imagem
                dataset_prefix = os.path.basename(base_dir.rstrip("/\\"))
                output_filename = os.path.join(class_output_dir, f"{dataset_prefix}_{filename}")
                cv2.imwrite(output_filename, processed_image)

            except Exception as e:
                print(f"Erro processando {filename}: {e}")
                continue

# --- Execução principal ---
if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for dataset_dir in RAW_DATA_DIRS:
        if not os.path.exists(dataset_dir):
            print(f"Aviso: Dataset não encontrado: {dataset_dir}")
            continue
        process_dataset(dataset_dir, PROCESSED_DATA_DIR)

    print(f"\n✅ Pré-processamento concluído. Todas as imagens processadas estão em: {PROCESSED_DATA_DIR}")
