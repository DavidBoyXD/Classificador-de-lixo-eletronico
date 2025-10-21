# -*- coding: utf-8 -*-

# Referências:
# TensorFlow/Keras ImageDataGenerator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# Keras Sequential Model: https://www.tensorflow.org/guide/keras/sequential_model
# Callbacks (EarlyStopping, ModelCheckpoint): https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_binary_classifier_model(input_shape):
    """
    Cria um modelo de CNN otimizado para classificação binária (2 classes).
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # Para classificação binária, uma única unidade com ativação sigmoid é o padrão.
        # A saída será um valor entre 0 e 1, representando a probabilidade de pertencer à classe '1'.
        Dense(1, activation='sigmoid')
    ])
    
    # Compila o modelo com 'binary_crossentropy', otimizado para problemas de 2 classes.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # --- Configurações para o Modelo de 2 Classes ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'processed_2_classes')
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'app', 'model')
    
    # Nomes de arquivo para o novo modelo
    NEW_MODEL_FILENAME = 'e_waste_classifier_2_classes_best.keras'
    NEW_CLASS_NAMES_FILENAME = 'class_names_2.txt'

    # As classes são lidas em ordem alfabética pelo ImageDataGenerator
    CLASS_NAMES = ["Desktop-PC", "Smartphone"]
    NUM_CLASSES = len(CLASS_NAMES)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32  # Reduzido para evitar problemas de memória
    EPOCHS = 250    # Aumentamos um pouco as épocas para o novo modelo

    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print(f"ERRO: O diretório de dados processados '{PROCESSED_DATA_DIR}' está vazio ou não existe.")
        print("Por favor, execute o script 'src/preprocess.py' primeiro.")
        exit()

    # --- Geradores de Dados ---
    # Como as imagens já foram pré-processadas (filtros aplicados), o gerador
    # só precisa normalizar os pixels (dividir por 255) e aplicar data augmentation.
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

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # O class_mode é 'binary' para problemas de 2 classes com loss 'binary_crossentropy'
    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    # --- Construção e Treinamento do Modelo ---
    model = create_binary_classifier_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    model.summary()

    # --- Callbacks ---
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_OUTPUT_DIR, NEW_MODEL_FILENAME),
        save_best_only=True,
        monitor='val_accuracy'
    )

    print("\nIniciando o treinamento do modelo de 2 classes...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("\nTreinamento concluído.")
    model_path = os.path.join(MODEL_OUTPUT_DIR, NEW_MODEL_FILENAME)
    print(f"O melhor modelo foi salvo em: {model_path}")

    # --- Salvando os nomes das classes ---
    class_names_path = os.path.join(MODEL_OUTPUT_DIR, NEW_CLASS_NAMES_FILENAME)
    with open(class_names_path, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    print(f"Nomes das classes salvos em: {class_names_path}")