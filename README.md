📱 Detecção de Smartphones com YOLOv8

Este projeto utiliza o modelo YOLOv8 para detectar smartphones em imagens.

O objetivo é identificar automaticamente a presença de smartphones em fotos — por exemplo, em ambientes como lixões — e destacar o objeto detectado com um retângulo colorido.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 Objetivo do Projeto

Detectar smartphones em imagens comuns.

Destacar automaticamente o objeto encontrado com uma borda colorida.

Gerar uma imagem final destacando o smartphone detectado.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📁 Estrutura do Projeto
project/

│

│  detect_smartphone.py     # Script para rodar a detecção

│

├── dataset/

│   ├── train/                   # Imagens de treino

│   ├── valid/                   # Imagens de validação

│   └── data.yaml                # Configuração do dataset

│

├── runs/                        # Saída do YOLO (modelos, logs, etc.)

│

├── test_images/                 # Imagens para testar o modelo

│

└── results_detected/            # Imagens com smartphones destacados
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ Requisitos

Antes de começar, você precisa ter instalado:

Python 3.10+

pip

Ambiente virtual venv (recomendado)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Configuração do Ambiente
1. Criar e ativar o ambiente virtual

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
venv\Scripts\activate

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. Instalar dependências

Com o ambiente virtual ativo, execute:

pip install ultralytics opencv-python
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🏋️‍♂️ Etapa 1 — Treinar o Modelo

Para treinar o modelo YOLOv8 com seu próprio dataset:

yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640

📁 Após o treinamento, os arquivos do modelo serão salvos automaticamente em:

runs/detect/train/weights/

O arquivo principal será:

best.pt

💡 Dica: Você pode ajustar o número de epochs conforme o desempenho desejado.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧪 Etapa 2 — Testar o Modelo

Após o treinamento, copie o arquivo best.pt para a raiz do projeto (ou mantenha-o em runs/detect/train/weights/).
Depois, adicione suas imagens de teste em:

test_images/

E execute o script:

python app/detect_smartphone.py

📸 Resultado

As imagens processadas com detecções serão salvas em:

results_detected/

Cada imagem que tiver um smartphone detectado será destacada com um retângulo verde e exibirá o nível de confiança da detecção.
Exemplo:

✅ Smartphone detectado! 

Resultado salvo em: results_detected/detected_imagem.jpg
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🗃️ Pastas Grandes no Drive

As pastas que contêm arquivos muito pesados estão disponíveis no Google Drive:

🔗 Link: https://drive.google.com/drive/folders/1FRKqSWGMn-sqSd9VE9FjX8e5pGWbbM1v?usp=drive_link

Baixe e coloque as pastas correspondentes na estrutura correta do projeto antes de rodar o treinamento.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
💡 Observações Importantes

Todos os comandos devem ser executados dentro do ambiente virtual venv.
Caso contrário, o pip e o YOLOv8 não funcionarão corretamente.

O modelo YOLOv8 é leve, mas requer uma GPU (opcional) para acelerar o treinamento.

O arquivo detect_smartphone.py já inclui o limiar de confiança configurável (CONFIDENCE_THRESHOLD).
