import os
import cv2
import yaml
from tqdm import tqdm

def apply_preprocessing_pipeline(image):
    """
    Aplica um pipeline de pré-processamento em uma imagem.
    1. Redimensiona para 224x224 pixels.
    2. Converte para escala de cinza.
    3. Aplica um filtro Gaussiano para suavizar.
    4. Aplica Equalização de Histograma para melhorar o contraste.
    """
    # 1. Redimensionamento
    resized_image = cv2.resize(image, (224, 224))
    # 2. Escala de Cinza
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # 3. Filtro Gaussiano
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # 4. Equalização de Histograma
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image

def process_dataset(base_dir, output_dir, class_names):
    """
    Processa o dataset original, aplicando o pré-processamento e organizando
    as imagens em subdiretórios de classe.
    """
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.isdir(img_dir):
            print(f"Diretório de imagens não encontrado para o split '{split}': {img_dir}")
            continue

        print(f"Processando o split: {split}...")
        
        image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc=f"Processando {split}"):
            img_path = os.path.join(img_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

            if not os.path.exists(label_path):
                print(f"Aviso: Arquivo de label não encontrado para {filename}. Pulando.")
                continue

            try:
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    
                    class_id = int(first_line.split()[0])
                    class_name = class_names[class_id]

                # Cria o diretório da classe se não existir
                class_output_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(class_output_dir, exist_ok=True)

                # Carrega e processa a imagem
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Aviso: Não foi possível ler a imagem {img_path}. Pulando.")
                    continue
                
                processed_image = apply_preprocessing_pipeline(image)
                
                # Salva a imagem processada
                output_filename = os.path.join(class_output_dir, filename)
                cv2.imwrite(output_filename, processed_image)

            except Exception as e:
                print(f"Erro ao processar o arquivo {filename}: {e}")


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
    DATA_YAML_PATH = os.path.join(RAW_DATA_DIR, 'data.yaml')

    # Cria o diretório de saída principal
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Carrega os nomes das classes do data.yaml
    try:
        with open(DATA_YAML_PATH, 'r') as file:
            data_config = yaml.safe_load(file)
        CLASS_NAMES = data_config['names']
        print("Nomes das classes carregados com sucesso.")
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRO: Não foi possível ler o arquivo de configuração do dataset. {e}")
        exit()

    # Processa o dataset
    process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES)

    print("\nPré-processamento concluído.")
    print(f"Imagens processadas e organizadas em: {PROCESSED_DATA_DIR}")