# Classificador de Lixo Eletrônico

Este projeto utiliza uma Rede Neural Convolucional (CNN) para classificar imagens de lixo eletrônico em diferentes categorias.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- `app/`: Contém a aplicação final e os modelos treinados.
- `dataset/`: Deve conter os dados brutos, processados e o arquivo `data.yaml`.
- `notebooks/`: Jupyter Notebooks para exploração e testes.
- `report/`: Relatórios e documentação do projeto.
- `src/`: Scripts fonte, como pré-processamento e treinamento.

## Como Executar o Treinamento

Siga os passos abaixo para configurar o ambiente e treinar o modelo.

### 1. Clone o Repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd lixo-eletronico-classificador
```

### 2. Crie e Ative um Ambiente Virtual

É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Crie o ambiente virtual
python3 -m venv .venv

# Ative o ambiente (Linux/macOS)
source .venv/bin/activate

# Ative o ambiente (Windows)
# .\venv\Scripts\activate
```

### 3. Instale as Dependências

Instale todas as bibliotecas necessárias a partir do arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Execute o Treinamento

Com o ambiente configurado, você pode iniciar o treinamento do modelo. O script irá ler os dados da pasta `dataset/processed`, treinar a CNN e salvar o melhor modelo em `app/model/`.

```bash
python src/train.py
```

**Nota:** Você pode ajustar hiperparâmetros como `BATCH_SIZE` e `EPOCHS` diretamente no arquivo `src/train.py` para adequar o treinamento à sua máquina.
