# -*- coding: utf-8 -*-

# Referência Geral para TensorFlow/Keras:
# Link: https://www.tensorflow.org/api_docs/python/tf/keras
# Referência Geral para OpenCV:
# Link: https://docs.opencv.org/4.x/

import os
import numpy as np
import tensorflow as tf
import cv2 # OpenCV para pré-processamento de imagens

class Classifier:
    """
    Classe que encapsula o modelo de classificação de lixo eletrônico.
    """
    def __init__(self, model_path: str, class_names_path: str):
        """
        Inicializa o classificador carregando o modelo e os nomes das classes.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {model_path}")
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Arquivo de nomes de classe não encontrado em: {class_names_path}")

        print("Carregando modelo...")
        self.model = tf.keras.models.load_model(model_path)
        print("Modelo carregado com sucesso.")

        print("Carregando nomes de classes...")
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"{len(self.class_names)} classes carregadas.")
        
        self.input_shape = self.model.input_shape[1:3]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Carrega e pré-processa uma imagem para que ela seja compatível com o modelo.
        Esta função DEVE ser um espelho do pipeline em 'preprocess.py'.
        """
        # 1. Carrega a imagem em escala de cinza
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem em: {image_path}")

        # 2. Redimensiona a imagem
        resized_image = cv2.resize(img, self.input_shape)

        # 3. Aplica Filtro Gaussiano
        blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

        # 4. Aplica Equalização de Histograma
        equalized_image = cv2.equalizeHist(blurred_image)

        # 5. Normaliza os pixels para o intervalo [0, 1] para o modelo
        normalized_image = equalized_image / 255.0

        # 6. Expande as dimensões para o formato do modelo (1, 224, 224, 1)
        final_image = np.expand_dims(normalized_image, axis=[0, -1])
        
        return final_image

    def classify_image(self, image_path: str) -> tuple[str, float]:
        """
        Executa o pipeline completo: pré-processa a imagem e retorna a classe e a confiança.

        Returns:
            tuple[str, float]: Uma tupla contendo o nome da classe prevista e a confiança (0.0 a 1.0).
        """
        processed_image = self.preprocess_image(image_path)
        
        # O modelo binário com sigmoid retorna um único valor entre 0 e 1.
        raw_prediction = self.model.predict(processed_image)[0][0]

        # Nossas classes são ["Desktop-PC", "Smartphone"] (ordem alfabética)
        # Se a predição for > 0.5, é a classe 1 (Smartphone)
        # Se for < 0.5, é a classe 0 (Desktop-PC)
        if raw_prediction > 0.5:
            predicted_class_name = self.class_names[1] # Smartphone
            confidence = raw_prediction
        else:
            predicted_class_name = self.class_names[0] # Desktop-PC
            confidence = 1 - raw_prediction
        
        return predicted_class_name, float(confidence)

if __name__ == '__main__':
    print("Executando o módulo classificador diretamente para teste...")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Aponta para os artefatos do modelo de 2 classes
    MODEL_PATH = os.path.join(BASE_DIR, 'app', 'model', 'e_waste_classifier_2_classes_best.keras')
    CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'app', 'model', 'class_names_2.txt')
    
    TEST_IMAGE_DIR = os.path.join(BASE_DIR, 'test_images')
    if not os.path.exists(TEST_IMAGE_DIR):
        os.makedirs(TEST_IMAGE_DIR)
        print(f"Diretório de teste criado em: {TEST_IMAGE_DIR}")
        print("Por favor, adicione uma imagem de lixo eletrônico neste diretório para testar.")
        exit()

    test_image_name = next((f for f in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(os.path.join(TEST_IMAGE_DIR, f))), None)

    if not test_image_name:
        print(f"Nenhuma imagem encontrada em {TEST_IMAGE_DIR} para testar.")
        exit()

    TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, test_image_name)

    try:
        classifier = Classifier(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH)
        print(f"\nClassificando a imagem: {TEST_IMAGE_PATH}")
        
        # A função agora retorna uma tupla (classe, confiança)
        predicted_class, confidence_score = classifier.classify_image(TEST_IMAGE_PATH)
        
        print(f"\n--- Resultado do Teste ---")
        print(f"A imagem '{test_image_name}' foi classificada como: {predicted_class}")
        print(f"Confiança: {confidence_score:.2%}")
        print("-------------------------")

    except FileNotFoundError as e:
        print(f"\nERRO: {e}")
        print(f"Verifique se o modelo treinado ('{os.path.basename(MODEL_PATH)}') e as classes (\'{os.path.basename(CLASS_NAMES_PATH)}\') existem no diretório 'app/model/'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o teste: {e}")
