# -*- coding: utf-8 -*-

# Referências:
# argparse (para argumentos de linha de comando): https://docs.python.org/3/library/argparse.html
# os (para manipulação de caminhos): https://docs.python.org/3/library/os.html

import os
import argparse
from src.classifier import Classifier
from src.database import init_db, log_classification, DB_PATH

# --- Constantes da Aplicação ---
CONFIDENCE_THRESHOLD = 0.80  # 80% de confiança mínima para considerar uma classificação válida.
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def get_image_paths(input_path: str) -> list[str]:
    """
    Verifica o caminho de entrada e retorna uma lista de caminhos de imagem válidos.
    - Se o caminho for um arquivo, retorna uma lista com esse arquivo.
    - Se o caminho for um diretório, retorna uma lista com todas as imagens válidas dentro dele.
    """
    image_paths = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(VALID_IMAGE_EXTENSIONS):
            image_paths.append(input_path)
        else:
            print(f"Aviso: O arquivo '{input_path}' não é uma imagem válida. Pulando.")
    elif os.path.isdir(input_path):
        print(f"Analisando o diretório: {input_path}")
        for filename in os.listdir(input_path):
            if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(input_path, filename))
    else:
        print(f"ERRO: O caminho '{input_path}' não é um arquivo ou diretório válido.")
    
    return image_paths

def main():
    """
    Função principal da aplicação.
    """
    # --- Configuração ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'app', 'model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'e_waste_classifier_2_classes_best.keras')
    CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names_2.txt')

    # --- Parser de Argumentos ---
    parser = argparse.ArgumentParser(description="Classificador de Lixo Eletrônico (Desktop-PC vs. Smartphone)")
    parser.add_argument("path", type=str, help="Caminho para a imagem ou pasta de imagens a ser classificada.")
    args = parser.parse_args()

    # --- Validação dos Arquivos Essenciais ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        print("ERRO: Modelo não encontrado! Execute 'src/train.py' primeiro.")
        return

    # --- Coleta dos Caminhos das Imagens ---
    image_paths_to_process = get_image_paths(args.path)
    if not image_paths_to_process:
        print("Nenhuma imagem para processar.")
        return

    # --- Inicialização do Banco de Dados e Classificador ---
    if not os.path.exists(DB_PATH):
        print("Banco de dados não encontrado. Inicializando...")
        init_db()
    
    try:
        print("Inicializando o classificador (isso pode levar um momento)...")
        classifier = Classifier(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH)
        print("Classificador pronto.")

        # --- Loop de Classificação ---
        print("-" * 50)
        for image_path in image_paths_to_process:
            image_name = os.path.basename(image_path)
            print(f"Processando: {image_name}...")
            
            predicted_class, confidence = classifier.classify_image(image_path)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"  -> Resultado: {predicted_class} (Confiança: {confidence:.2%})")
                log_classification(
                    image_name=image_name,
                    image_path=os.path.abspath(image_path),
                    predicted_class=predicted_class
                )
            else:
                print(f"  -> Resultado: Não identificado (Confiança muito baixa: {confidence:.2%})")
        print("-" * 50)
        print("\nProcesso concluído.")

    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")

if __name__ == '__main__':
    main()