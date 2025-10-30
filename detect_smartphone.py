# -*- coding: utf-8 -*-
"""
Detecta smartphones em uma imagem usando o modelo YOLOv8 treinado.
Destaca os smartphones detectados e salva a imagem final.
"""

import os
from ultralytics import YOLO
import cv2

# --- Configurações Principais ---
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Caminho do modelo treinado
INPUT_PATH = "test_images"  # Pasta com as imagens a testar
OUTPUT_PATH = "results_detected"  # Pasta onde salvará as imagens com detecções
CONFIDENCE_THRESHOLD = 0.4  # Confiança mínima (0.0 a 1.0)

# --- Garante que as pastas existem ---
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
# --- Cria as pastas test_images e results_detected se ainda não existirem ---

# --- Carrega o modelo ---
print(" Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)
print(" Modelo carregado com sucesso!\n")
# --- Inicializa o modelo YOLO com o arquivo best.pt ---

# --- Carrega as imagens para testar ---
image_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# --- Cria uma lista de todas as imagens da pasta test_images ---


if not image_files:
    print(f"Nenhuma imagem encontrada em '{INPUT_PATH}'.")
    print("Adicione imagens para testar.")
    exit()
    # --- Encerra o programa se não houver imagens na pasta de entrada ---

# --- Loop de detecção ---
for image_name in image_files:
    image_path = os.path.join(INPUT_PATH, image_name)
    print(f" Processando: {image_name}")

    # --- Executa o modelo ---
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    """
    Passa a imagem pelo modelo YOLO.
    Retorna uma lista de objetos results, contendo as caixas, classes e confianças detectadas.
    """

    # --- Carrega imagem original ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar {image_name}")
        continue

    # --- Interpreta as detecções ---
    detected = False

    for result in results:
        boxes = result.boxes.xyxy  # Coordenadas (x1, y1, x2, y2)
        confidences = result.boxes.conf # Confiança de cada detecção
        class_ids = result.boxes.cls # ID da classe (ex: 0 = smartphone)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Desenha retângulo e o texto
            color = (0, 255, 0)  # Verde para smartphone
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                image,
                f"Smartphone {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            detected = True

    # --- Salva a imagem final ---
    output_file = os.path.join(OUTPUT_PATH, f"detected_{image_name}")
    cv2.imwrite(output_file, image)

    # --- Salvar log da detecção ---
    log_path = os.path.join(OUTPUT_PATH, "detections_log.txt")
    with open(log_path, "a", encoding="utf-8") as log:
        if detected:
            log.write(f"\n {image_name} → {len(detections_info)} smartphone(s) detectado(s):\n")
            for info in detections_info:
                log.write(f"   - {info}\n")
        else:
            log.write(f"\n {image_name} → Nenhum smartphone detectado (confiança mínima {CONFIDENCE_THRESHOLD})\n")


    # --- Mostra o status da detecção ---
    if detected:
        print(f" Smartphone detectado! Resultado salvo em: {output_file}")
    else:
        print(f" Nenhum smartphone encontrado em: {image_name}")

print("\n Detecção concluída!")
print(f"Imagens processadas salvas em: {OUTPUT_PATH}")
# --- Mensagem final indicando sucesso do processo. ---