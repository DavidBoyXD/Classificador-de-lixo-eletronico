
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yaml

def create_model(input_shape, num_classes):
    """
    Cria um modelo de Rede Neural Convolucional (CNN) com a API Keras.
    """
    model = Sequential([
        Input(shape=input_shape),
        # Camada Convolucional: extrai características (bordas, texturas)
        # 32 filtros, kernel 3x3, função de ativação ReLU
        Conv2D(32, (3, 3), activation='relu'),
        # Camada de Pooling: Reduz a dimensionalidade (down-sampling)
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Achatamento: Transforma o mapa de características 2D em um vetor 1D
        Flatten(),
        
        # Camada Densa (Totalmente Conectada): Camada de classificação
        Dense(128, activation='relu'),
        # Dropout: Técnica de regularização para prevenir overfitting
        Dropout(0.5),
        
        # Camada de Saída: O número de neurônios é igual ao número de classes
        # Ativação Softmax para problemas de classificação multiclasse
        Dense(num_classes, activation='softmax')
    ])
    
    # Compila o modelo com o otimizador Adam, função de perda para classificação
    # e métrica de acurácia.
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # --- Configurações ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
    DATA_YAML_PATH = os.path.join(BASE_DIR, 'dataset', 'data.yaml')
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'app', 'model')
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32 # Número de imagens a serem processadas em cada lote
    EPOCHS = 20     # Número de vezes que o modelo verá todo o dataset de treino

    # --- Carregar Configurações do Dataset ---
    try:
        with open(DATA_YAML_PATH, 'r') as file:
            data_config = yaml.safe_load(file)
        NUM_CLASSES = data_config['nc']
        CLASS_NAMES = data_config['names']
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRO: Não foi possível ler o arquivo de configuração do dataset. {e}")
        exit()

    # --- Geradores de Dados ---
    # Cria um gerador de dados para o conjunto de treino com Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Cria um gerador de dados para o conjunto de validação (apenas reescala)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # --- Construção e Treinamento do Modelo ---
    # O input_shape é (altura, largura, canais). Como é escala de cinza, temos 1 canal.
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=NUM_CLASSES)
    model.summary() # Exibe um resumo da arquitetura do modelo

    # Callbacks para otimizar o treinamento
    # EarlyStopping: para o treinamento se a performance não melhorar
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    # ModelCheckpoint: salva o melhor modelo encontrado durante o treinamento
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_OUTPUT_DIR, 'e_waste_classifier_best.keras'),
        save_best_only=True,
        monitor='val_accuracy'
    )

    print("\nIniciando o treinamento do modelo...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("\nTreinamento concluído.")
    print(f"O melhor modelo foi salvo em: {os.path.join(MODEL_OUTPUT_DIR, 'e_waste_classifier_best.keras')}")

    # --- Salvando os nomes das classes ---
    # Salva os nomes das classes em um arquivo para uso posterior na aplicação
    class_names_path = os.path.join(MODEL_OUTPUT_DIR, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    print(f"Nomes das classes salvos em: {class_names_path}")
