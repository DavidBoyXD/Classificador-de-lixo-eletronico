# -*- coding: utf-8 -*-
"""
Detecta smartphones em imagens usando o modelo YOLOv8.
Aplica pré-processamento (Gaussian Blur + Laplacian)
para realçar bordas e reduzir ruído antes da inferência.
Gera um log CSV com as detecções.
"""

import cv2
import os
import csv
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
MODEL_PATH = "runs/detect/train14/weights/best.pt"   # Caminho do modelo YOLO treinado
INPUT_PATH = "test_images"                           # Pasta com imagens para teste
OUTPUT_PATH = "results_detected"                     # Pasta para salvar imagens detectadas
LOG_FILE = os.path.join(OUTPUT_PATH, "detections_log.csv")  # Caminho do log CSV
CONFIDENCE_THRESHOLD = 0.3                        # Confiança mínima 

# --- GARANTE QUE AS PASTAS EXISTAM ---
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- CARREGA O MODELO ---
print(" Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)
print(" Modelo carregado com sucesso!\n")

# --- INICIALIZA O LOG CSV ---
with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Imagem", "Classe", "Confiança", "x1", "y1", "x2", "y2"])

# --- LISTA AS IMAGENS PARA PROCESSAR ---
image_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f" Nenhuma imagem encontrada em '{INPUT_PATH}'.")
    print("Adicione imagens para testar e execute novamente.")
    exit()

# --- LOOP PRINCIPAL ---
for image_name in image_files:
    image_path = os.path.join(INPUT_PATH, image_name)
    print(f" Processando: {image_name}")

    # --- CARREGAR IMAGEM ---
    image = cv2.imread(image_path)
    if image is None:
        print(f" Erro ao carregar {image_name}")
        continue

    # --- PRÉ-PROCESSAMENTO ---
    # Gaussian Blur (suaviza ruídos da imagem)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Conversão para tons de cinza
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Filtro Laplaciano (realce de bordas)
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = cv2.convertScaleAbs(laplace)

    # Combina imagem suavizada + bordas realçadas
    image_preprocessed = cv2.addWeighted(blurred, 0.8, cv2.cvtColor(laplace, cv2.COLOR_GRAY2BGR), 0.4, 0)

    # --- INFERÊNCIA YOLO ---
    results = model.predict(source=image_preprocessed, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detected = False

    # --- PROCESSAR RESULTADOS ---
    for result in results:
        boxes = result.boxes.xyxy  # Coordenadas (x1, y1, x2, y2)
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            confidence = float(conf)

            # --- DESENHAR RETÂNGULO ---
            color = (0, 255, 0)  # Verde = smartphone
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                image,
                f"Smartphone {confidence:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            detected = True

            # --- REGISTRAR NO LOG ---
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([image_name, "Smartphone", confidence, x1, y1, x2, y2])

    # --- SALVAR RESULTADO ---
    output_file = os.path.join(OUTPUT_PATH, f"detected_{image_name}")
    cv2.imwrite(output_file, image)

    if detected:
        print(f" Smartphone detectado! Resultado salvo em: {output_file}")
    else:
        print(f" Nenhum smartphone encontrado em: {image_name}")

print("\nDetecção concluída!")
print(f"Imagens processadas salvas em: {OUTPUT_PATH}")
print(f"Log de detecções salvo em: {LOG_FILE}")
