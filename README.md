
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

## Ambiente de Desenvolvimento Recomendado: Windows + WSL

Para garantir que o projeto funcione sem erros de permissão de arquivo entre Windows e Linux, a forma correta e profissional de trabalhar é usando o Visual Studio Code conectado diretamente ao seu ambiente WSL.

### Como Funciona?

Ao invés de abrir os arquivos do Linux no VS Code "rodando" no Windows, nós faremos o VS Code "rodar" dentro do próprio Linux. A janela que você vê no Windows funcionará como um controle remoto. Isso é feito através da extensão oficial da Microsoft e evita todos os conflitos de permissão.

### Passos para Conexão

1.  **Instale a Extensão:** Dentro do VS Code, vá ao menu de Extensões (ícone de blocos na lateral), procure por `WSL` e instale a extensão criada pela Microsoft.

2.  **Abra o Projeto (Escolha uma opção):**

    *   **Opção A (Recomendada): Via Terminal**
        1.  Abra seu terminal do Ubuntu (pelo Menu Iniciar do Windows).
        2.  Navegue até a pasta onde você clonou o projeto (ex: `cd /caminho/ate/Classificador-de-lixo-eletronico`).
        3.  Dentro da pasta, digite o comando:
            ```bash
            code .
            ```
        4.  Isso abrirá uma nova janela do VS Code, já conectada ao WSL.

    *   **Opção B: Via Interface do VS Code**
        1.  Abra o VS Code normalmente no Windows.
        2.  Clique no botão verde no canto inferior esquerdo da janela (geralmente mostra `><`).
        3.  No menu que aparece no topo, selecione **"Conectar ao WSL"** (ou *Connect to WSL*).
        4.  Uma nova janela do VS Code será aberta, agora conectada ao WSL.
        5.  Nesta nova janela, vá em `Arquivo > Abrir Pasta...` (`File > Open Folder...`) e navegue até a pasta do seu projeto dentro do sistema de arquivos do Linux (ex: `/home/seu_usuario/Classificador-de-lixo-eletronico`).

3.  **Verificação:** Em ambos os casos, confirme que a conexão foi bem-sucedida olhando para o canto inferior esquerdo do VS Code. Deverá haver um botão verde indicando **"WSL: Ubuntu"** (ou o nome da sua distribuição). A partir deste ponto, todo comando executado no terminal integrado do VS Code (`Ctrl+'`) será executado dentro do Linux, como esperado.

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
