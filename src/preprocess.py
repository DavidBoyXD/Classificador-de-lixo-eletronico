# -*- coding: utf-8 -*- 

# Referência Geral para OpenCV (cv2): 
# Link: https://docs.opencv.org/4.x/ 
# Referência Geral para PyYAML: 
# Link: https://pyyaml.org/wiki/PyYAMLDocumentation

import os
import cv2
import yaml
from tqdm import tqdm

# --- Configuração das Classes Alvo ---
TARGET_CLASSES = ["Smartphone", "Desktop-PC"]

def apply_preprocessing_pipeline(image):
    """
    Aplica o pipeline de pré-processamento com filtros clássicos.
    1. Converte para escala de cinza.
    2. Redimensiona para 224x224 pixels.
    3. Aplica Filtro Gaussiano para suavizar a imagem e reduzir ruídos.
    4. Aplica Equalização de Histograma para melhorar o contraste.
    """
    # 1. Garante que a imagem está em escala de cinza
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 2. Redimensionamento
    # Referência: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    resized_image = cv2.resize(gray_image, (224, 224))

    # 3. Filtro Gaussiano (kernel 5x5)
    # Referência: https://docs.opencv.org/4.x/d4/d13/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # 4. Equalização de Histograma
    # Referência: https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e
    equalized_image = cv2.equalizeHist(blurred_image)
    
    return equalized_image

def process_dataset(base_dir, output_dir, all_class_names):
    """
    Processa o dataset, selecionando apenas as classes alvo, aplicando o pré-processamento
    e salvando na nova estrutura de diretórios.
    """
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.isdir(img_dir):
            continue

        print(f"Processando o split: {split}...")
        image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc=f"Processando {split}"):
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
            if not os.path.exists(label_path):
                continue

            try:
                with open(label_path, 'r') as f:
                    class_id = int(f.readline().strip().split()[0])
                class_name = all_class_names[class_id]

                if class_name not in TARGET_CLASSES:
                    continue
                
                class_output_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(class_output_dir, exist_ok=True)

                img_path = os.path.join(img_dir, filename)
                # Carrega a imagem já em escala de cinza para otimizar
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                processed_image = apply_preprocessing_pipeline(image)
                
                output_filename = os.path.join(class_output_dir, filename)
                cv2.imwrite(output_filename, processed_image)

            except Exception:
                pass

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'processed_2_classes')
    DATA_YAML_PATH = os.path.join(RAW_DATA_DIR, 'data.yaml')

    print(f"Diretório de saída será: {PROCESSED_DATA_DIR}")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        with open(DATA_YAML_PATH, 'r') as file:
            data_config = yaml.safe_load(file)
        ALL_CLASS_NAMES = data_config['names']
        print(f"Classes alvo para o novo dataset: {TARGET_CLASSES}")
    except Exception as e:
        print(f"ERRO: Não foi possível ler o arquivo de configuração do dataset. {e}")
        exit()

    process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR, ALL_CLASS_NAMES)

    print("\nPré-processamento para 2 classes (com filtros) concluído.")
    print(f"Imagens processadas e organizadas em: {PROCESSED_DATA_DIR}")
