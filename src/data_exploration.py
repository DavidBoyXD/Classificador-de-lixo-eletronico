
import os
import cv2
import numpy as np
import yaml

def create_and_save_samples_grid(dataset_path, data_config, output_path, num_images=3, img_size=(200, 200)):
    """
    Lê imagens de amostra dos conjuntos de treino, validação e teste,
    cria uma grade visual e a salva como um único arquivo de imagem.
    """
    all_labeled_rows = []

    for set_type in ['train', 'val', 'test']:
        # Constrói o caminho para a pasta de imagens do conjunto (treino, val, teste)
        image_folder_relative = data_config.get(set_type)
        if not image_folder_relative:
            print(f"Aviso: Chave '{set_type}' não encontrada em data.yaml. Pulando.")
            continue
        
        # O caminho em data.yaml (ex: ../train/images) está incorreto para nossa estrutura.
        # Vamos remover o '../' para construir o caminho correto dentro da pasta 'dataset'.
        if image_folder_relative.startswith('../'):
            image_folder_relative = image_folder_relative[3:]

        image_folder_absolute = os.path.abspath(os.path.join(dataset_path, image_folder_relative))

        try:
            image_files = [f for f in os.listdir(image_folder_absolute) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_images]
        except FileNotFoundError:
            print(f"Aviso: Diretório não encontrado: {image_folder_absolute}. Pulando conjunto '{set_type}'.")
            continue

        image_row = []
        for img_file in image_files:
            img_path = os.path.join(image_folder_absolute, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                image_row.append(img)
            else:
                # Adiciona uma imagem preta caso o carregamento falhe
                image_row.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))

        # Garante que a linha tenha 'num_images' imagens, preenchendo com placeholders se necessário
        while len(image_row) < num_images:
            image_row.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))

        # Combina as imagens da linha horizontalmente
        combined_row = np.hstack(image_row)

        # Cria um rótulo para a linha
        label_height = 50
        row_label = np.zeros((label_height, combined_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(row_label, f"Amostras: {set_type.upper()}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Adiciona o rótulo e a linha de imagens à lista final
        all_labeled_rows.append(row_label)
        all_labeled_rows.append(combined_row)

    if not all_labeled_rows:
        print("Nenhuma imagem foi processada. A imagem de saída não será gerada.")
        return

    # Combina todas as linhas rotuladas verticalmente
    final_grid = np.vstack(all_labeled_rows)
    
    # Salva a imagem final
    cv2.imwrite(output_path, final_grid)
    print(f"Grade de amostras salva em: {output_path}")

if __name__ == "__main__":
    # Constrói os caminhos de forma dinâmica a partir da localização do script
    # __file__ -> src/data_exploration.py
    # os.path.dirname(__file__) -> src/
    # os.path.dirname(os.path.dirname(__file__)) -> /lixo-eletronico-classifier/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
    DATA_YAML_PATH = os.path.join(DATASET_PATH, 'data.yaml')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'report', 'figures')
    
    # Garante que o diretório de saída exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with open(DATA_YAML_PATH, 'r') as file:
            data_config = yaml.safe_load(file)
        
        output_image_path = os.path.join(OUTPUT_DIR, 'amostras_dataset.png')
        
        create_and_save_samples_grid(DATASET_PATH, data_config, output_image_path)

    except FileNotFoundError:
        print(f"ERRO: Arquivo 'data.yaml' não encontrado em '{DATA_YAML_PATH}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
