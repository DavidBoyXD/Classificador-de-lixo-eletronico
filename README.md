
# Classificador de Lixo Eletrônico (Smartphones vs. Desktops)

Este projeto é uma aplicação de linha de comando (CLI) que utiliza Visão Computacional e Deep Learning para classificar imagens de lixo eletrônico em duas categorias: **Desktop-PC** e **Smartphone**.

## Features

- **Classificação Binária:** Focado em duas das classes mais comuns de lixo eletrônico para maior precisão.
- **Pipeline de Pré-processamento:** Utiliza filtros clássicos do OpenCV (Filtro Gaussiano e Equalização de Histograma) para tratar as imagens antes do treinamento.
- **Limite de Confiança:** Apenas classificações com confiança acima de 80% são consideradas válidas, evitando palpites incertos.
- **Análise em Lote:** Processa tanto imagens individuais quanto pastas inteiras de imagens de uma só vez.
- **Banco de Dados:** Salva cada classificação bem-sucedida em um banco de dados SQLite para análise e rastreabilidade, utilizando SQLAlchemy para maior portabilidade.
- **Estrutura Modular:** O código é organizado em módulos para fácil manutenção e entendimento (`database.py`, `classifier.py`).

---

## Ambiente de Desenvolvimento (Windows)

Para evitar problemas de permissão entre o Windows e o Linux, a forma **altamente recomendada** de trabalhar neste projeto é usando o **WSL (Windows Subsystem for Linux)** com o **Visual Studio Code**.

1.  **Instale a Extensão:** No VS Code, instale a extensão oficial da Microsoft chamada **[Remote - WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)**.

2.  **Abra o Projeto no WSL:**
    - Abra seu terminal Ubuntu (ou outra distro WSL).
    - Navegue até a pasta onde você clonou este projeto.
    - Dentro da pasta, execute o comando: `code .`

    Isso abrirá o VS Code conectado diretamente ao ambiente Linux. Você saberá que funcionou se vir **"WSL: Ubuntu"** no canto inferior esquerdo do VS Code.

---

## Como Configurar e Usar

Siga os passos abaixo para configurar e executar o projeto.

### Passo 0: Pré-requisitos

- Python 3.8+
- **Download do Dataset:** O dataset original é necessário. 
    1. Baixe-o em: **[Balanced E-Waste Dataset no Roboflow](https://universe.roboflow.com/david-andrew-e1p1t/balanced-e-waste-dataset-77kuk/dataset/1)**.
    2. Escolha o formato "YOLOv5 PyTorch" e faça o download do arquivo ZIP.
    3. Descompacte o conteúdo do ZIP dentro da pasta `dataset/` do projeto. A estrutura final deve ter as pastas `dataset/train`, `dataset/valid`, `dataset/test`.

### Passo 1: Configuração do Ambiente

1.  **Clone o repositório** (ou baixe e extraia os arquivos).

2.  **Crie um ambiente virtual.** Abra o terminal (dentro do VS Code já conectado ao WSL) e execute:
    ```bash
    python3 -m venv .venv
    ```

3.  **Ative o ambiente virtual:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Instale as dependências.** Com o ambiente ativado, instale todas as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```

### Passo 2: Pré-processamento dos Dados

Antes de treinar, você precisa criar o dataset focado apenas nas duas classes de interesse. Para isso, execute o script de pré-processamento:

```bash
python3 src/preprocess.py
```

Isso irá criar a pasta `dataset/processed_2_classes` com as imagens já filtradas e tratadas.

### Passo 3: Treinamento do Modelo

**Atenção:** O treinamento de redes neurais é um processo computacionalmente intensivo e é **altamente recomendado executá-lo em uma máquina com uma GPU NVIDIA compatível com CUDA.**

Para iniciar o treinamento, execute:

```bash
python3 src/train.py
```

Ao final, o melhor modelo será salvo como `app/model/e_waste_classifier_2_classes_best.keras`.

### Passo 4: Classificar Imagens

Com o modelo treinado, você pode finalmente usar a aplicação principal para classificar novas imagens.

Passe o caminho de um **arquivo de imagem** ou de uma **pasta com imagens** como argumento.

**Exemplo com arquivo único:**
```bash
python3 app.py "/mnt/c/Users/SeuUsuario/Downloads/foto_celular.png"
```

**Exemplo com uma pasta:**
```bash
python3 app.py "/mnt/c/Users/SeuUsuario/Documents/minhas_imagens"
```

*Nota sobre caminhos:* Lembre-se que, dentro do WSL, o seu disco `C:` do Windows é acessado via `/mnt/c/`.

Os resultados serão exibidos no terminal. As classificações com alta confiança serão salvas no banco de dados.

---

## Banco de Dados

Os resultados de cada classificação bem-sucedida são armazenados no arquivo `app/ewaste_log.db`, na tabela `classifications`.

As colunas da tabela são:
- `id`: Identificador único da classificação.
- `timestamp`: Data e hora em que a classificação foi feita.
- `image_name`: Nome do arquivo da imagem.
- `image_path`: Caminho completo para a imagem classificada.
- `predicted_class`: A classe que o modelo previu (Desktop-PC ou Smartphone).

Você pode inspecionar este arquivo usando qualquer ferramenta compatível com SQLite, como o [DB Browser for SQLite](https://sqlitebrowser.org/).
