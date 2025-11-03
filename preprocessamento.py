# -*- coding: utf-8 -*-
"""
Pré-processamento adaptativo seguro para detecção de smartphones em meio a e-waste.
Cria uma cópia processada do dataset original em 'dataset_processado/', sem sobrescrever arquivos.
Ajusta automaticamente os filtros conforme contraste, brilho e ruído da imagem.
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

# --- CONFIGURAÇÕES ---
ORIGINAL_DATASET = "dataset/MobilePhone"
PROCESSED_DATASET = "dataset_processado/MobilePhone"
SUBFOLDERS = ["train/images", "valid/images", "test/images"]

# --- FUNÇÕES AUXILIARES ---

def analyze_image_properties(image):
    """Analisa contraste e brilho médio da imagem."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return brightness, contrast


def adaptive_preprocessing(image):
    """Aplica filtros adaptativos conforme contraste e brilho da imagem."""
    brightness, contrast = analyze_image_properties(image)

    # Ajuste adaptativo de parâmetros
    bilateral_strength = 9 if contrast < 35 else 5
    laplace_strength = 0.5 if contrast < 40 else 0.2
    clahe_clip = 3.0 if brightness < 100 else 2.0

    # 1️⃣ Filtro bilateral: suaviza ruído preservando bordas
    filtered = cv2.bilateralFilter(image, d=bilateral_strength, sigmaColor=75, sigmaSpace=75)

    # 2️⃣ CLAHE: realce de contraste local
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 3️⃣ Laplaciano: realce adaptativo de bordas
    laplace = cv2.Laplacian(equalized, cv2.CV_64F)
    laplace = cv2.convertScaleAbs(laplace)

    # 4️⃣ Combinação ponderada (equilíbrio entre suavização e bordas)
    combined = cv2.addWeighted(equalized, 1 - laplace_strength, laplace, laplace_strength, 0)

    # 5️⃣ Normalização global
    normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)

    # 6️⃣ Converter para BGR (necessário para o YOLO)
    final = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return final


def copy_dataset_structure():
    """Cria a estrutura de pastas no dataset processado."""
    if os.path.exists(PROCESSED_DATASET):
        print(f"⚠️ A pasta '{PROCESSED_DATASET}' já existe — imagens serão atualizadas.")
    else:
        print("🗂️ Criando cópia da estrutura do dataset...")
        for sub in SUBFOLDERS:
            src_folder = os.path.join(ORIGINAL_DATASET, sub)
            dst_folder = os.path.join(PROCESSED_DATASET, sub)
            os.makedirs(dst_folder, exist_ok=True)

            # Também copiar pastas de labels
            label_folder = sub.replace("images", "labels")
            src_label_folder = os.path.join(ORIGINAL_DATASET, label_folder)
            dst_label_folder = os.path.join(PROCESSED_DATASET, label_folder)
            os.makedirs(dst_label_folder, exist_ok=True)

            # Copia arquivos de label
            if os.path.exists(src_label_folder):
                for label_file in os.listdir(src_label_folder):
                    if label_file.endswith(".txt"):
                        shutil.copy(
                            os.path.join(src_label_folder, label_file),
                            os.path.join(dst_label_folder, label_file)
                        )


def preprocess_folder(subfolder):
    """Aplica pré-processamento adaptativo e salva as imagens processadas."""
    src_path = os.path.join(ORIGINAL_DATASET, subfolder)
    dst_path = os.path.join(PROCESSED_DATASET, subfolder)

    image_files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(image_files, desc=f"Processando {subfolder}"):
        src_img_path = os.path.join(src_path, img_name)
        dst_img_path = os.path.join(dst_path, img_name)

        image = cv2.imread(src_img_path)
        if image is None:
            continue

        processed = adaptive_preprocessing(image)
        cv2.imwrite(dst_img_path, processed)


if __name__ == "__main__":
    print("🔹 Iniciando pré-processamento adaptativo seguro para e-waste...\n")
    copy_dataset_structure()

    for sub in SUBFOLDERS:
        preprocess_folder(sub)

    print("\n✅ Pré-processamento concluído com sucesso!")
    print(f"📁 Dataset processado salvo em: {PROCESSED_DATASET}")
