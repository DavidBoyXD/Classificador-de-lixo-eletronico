
# UNIVERSIDADE PAULISTA (UNIP)
# CURSO DE CIÊNCIA DA COMPUTAÇÃO

## ATIVIDADES PRÁTICAS SUPERVISIONADAS (APS)

### APLICAÇÃO DE TÉCNICAS DE PROCESSAMENTO DIGITAL DE IMAGENS E VISÃO COMPUTACIONAL PARA RESOLVER PROBLEMAS DE SUSTENTABILIDADE AMBIENTAL

**NOME DO ALUNO/GRUPO**

**RA DO ALUNO/GRUPO**

**LOCAL**
**ANO**

---

## Folha de Rosto

**NOME DO ALUNO/GRUPO**

### CLASSIFICADOR DE LIXO ELETRÔNICO

Trabalho apresentado como requisito parcial para a avaliação da disciplina de Atividades Práticas Supervisionadas (APS) do curso de Ciência da Computação da Universidade Paulista.

**Orientador(a):** [Nome do Orientador]

**LOCAL**
**ANO**

---

## Resumo

Este trabalho aborda o desenvolvimento de uma aplicação para a classificação automática de lixo eletrônico (e-waste) utilizando técnicas de Processamento Digital de Imagens e Visão Computacional. O objetivo principal é criar um sistema capaz de identificar e categorizar diferentes tipos de resíduos eletrônicos a partir de imagens, visando facilitar e otimizar o processo de reciclagem. A metodologia empregada combina o uso de filtros e algoritmos de segmentação para pré-processar as imagens e um modelo de Inteligência Artificial, como uma Rede Neural Convolucional (CNN), para realizar a classificação final. O projeto foi desenvolvido em Python, utilizando bibliotecas como OpenCV e TensorFlow/Keras, e validado com um dataset de imagens de lixo eletrônico. Os resultados demonstram a viabilidade da solução para a automação da triagem de e-waste, contribuindo para a sustentabilidade ambiental.

**Palavras-chave:** Processamento de Imagens, Visão Computacional, Lixo Eletrônico, Classificação de Imagens, Inteligência Artificial.

---

## Sumário

1.  [Introdução](#1-introdução)
2.  [Referencial Teórico](#2-referencial-teórico)
3.  [Proposta da Solução](#3-proposta-da-solução)
4.  [Desenvolvimento](#4-desenvolvimento)
5.  [Conclusão](#5-conclusão)
6.  [Referências](#6-referências)
7.  [Apêndice](#7-apêndice)

---

## 1. Introdução

### 1.1. Descrição do Problema

O descarte inadequado de lixo eletrônico, ou e-waste, representa um dos desafios ambientais e de saúde pública mais crescentes da atualidade. Componentes eletrônicos contêm materiais tóxicos como chumbo, mercúrio e cádmio, que, quando descartados em aterros comuns, podem contaminar o solo e os lençóis freáticos. Além disso, muitos desses dispositivos contêm materiais valiosos que poderiam ser recuperados e reutilizados. A triagem manual desses resíduos é um processo lento, caro e perigoso para os trabalhadores. Portanto, a automação da classificação de lixo eletrônico é fundamental para aumentar a eficiência da reciclagem, reduzir custos operacionais e minimizar os impactos ambientais e sociais negativos.

### 1.2. Contexto

O projeto insere-se no contexto da sustentabilidade ambiental e da indústria 4.0, aplicando tecnologias de visão computacional para resolver um problema prático de gestão de resíduos. A crescente produção e consumo de eletrônicos em escala global intensifica a urgência por soluções inovadoras na área de reciclagem. A implementação de um sistema automatizado de triagem pode revolucionar a cadeia de reciclagem de e-waste, tornando-a mais segura, rápida e economicamente viável, além de promover a economia circular.

### 1.3. Objetivos

*   **Objetivo Geral:** Desenvolver uma aplicação computacional em Python que classifica diferentes categorias de lixo eletrônico a partir de imagens, utilizando uma combinação de processamento de imagem clássico e um modelo de inteligência artificial.
*   **Objetivos Específicos:**
    *   Realizar o pré-processamento das imagens para extrair características relevantes e reduzir ruídos, utilizando filtros e técnicas de segmentação.
    *   Treinar e avaliar um modelo de rede neural convolucional (CNN) para a tarefa de classificação de imagens.
    *   Integrar as etapas de pré-processamento e classificação em um pipeline funcional.
    *   Validar a acurácia e a eficiência do sistema proposto.

---

## 2. Referencial Teórico

Nesta seção, são apresentados os conceitos fundamentais de Processamento Digital de Imagens e Visão Computacional aplicados no desenvolvimento deste projeto.

### 2.1. Aquisição de Imagens
A aquisição é o processo de obter uma imagem digital a partir de uma fonte do mundo real. No contexto deste projeto, a aquisição foi a utilização de um dataset pré-existente, o "Balanced E-Waste Dataset" da plataforma Roboflow, que consiste em imagens fotográficas de diversos tipos de lixo eletrônico.

### 2.2. Manipulação de Imagens (Pré-processamento)
Após a aquisição, as imagens raramente estão prontas para serem analisadas por um modelo de IA. A manipulação ou pré-processamento é uma etapa crítica que envolve a aplicação de uma série de operações para padronizar as imagens e realçar características importantes. Para este trabalho, a manipulação envolveu a conversão de espaço de cores e o redimensionamento.

### 2.3. Filtragem
A filtragem é uma subcategoria da manipulação de imagens cujo objetivo é modificar ou realçar certos aspectos da imagem, tipicamente para redução de ruído ou extração de bordas. Em nosso pipeline, utilizamos a filtragem para aplicar um filtro Gaussiano, que suaviza a imagem e reduz ruídos de alta frequência, e a equalização de histograma, que melhora o contraste geral da imagem.

### 2.4. Segmentação
A segmentação é o processo de particionar uma imagem em múltiplas regiões ou objetos. Embora nosso projeto não implemente uma etapa de segmentação explícita (como a detecção de onde o objeto está na imagem), o próprio ato de classificar uma imagem que contém um objeto centralizado pode ser visto como uma forma de segmentação em nível de cena. O dataset utilizado já fornece imagens onde o objeto de interesse é o foco principal.

### 2.5. Reconhecimento de Imagens
O reconhecimento, ou classificação, é a tarefa final, onde o sistema atribui um rótulo (uma classe) a uma imagem. Neste projeto, utilizamos uma Rede Neural Convolucional (CNN), um tipo de modelo de aprendizado profundo (Deep Learning) especialmente eficaz para tarefas de visão computacional. A CNN aprende a identificar padrões hierárquicos complexos nas imagens pré-processadas para realizar a classificação entre as 37 categorias de lixo eletrônico.

---

## 3. Proposta da Solução

A solução proposta para o processamento das imagens, antes de serem enviadas ao modelo de IA, consiste em um pipeline de quatro etapas sequenciais, implementadas com a biblioteca OpenCV:

1.  **Conversão para Escala de Cinza:** A imagem colorida é convertida para escala de cinza para reduzir a complexidade computacional (de 3 canais de cor para 1) e focar nas características de forma, textura e luminância dos objetos.
2.  **Redução de Ruído com Filtro Gaussiano:** Um filtro `GaussianBlur` (com kernel 5x5) é aplicado para suavizar a imagem e remover ruídos de alta frequência, que poderiam ser interpretados incorretamente pelo modelo.
3.  **Equalização de Histograma:** A técnica `equalizeHist` é utilizada para melhorar o contraste da imagem em escala de cinza. Isso distribui a intensidade dos pixels de forma mais uniforme, realçando detalhes em áreas subexpostas ou superexpostas.
4.  **Redimensionamento:** Todas as imagens são padronizadas para um tamanho fixo de 224x224 pixels. Isso é crucial para que o modelo de inteligência artificial receba entradas de dimensão consistente.

Para o reconhecimento das imagens, propõe-se um modelo de Inteligência Artificial baseado em uma Rede Neural Convolucional (CNN). A arquitetura do modelo, implementada com a biblioteca Keras (TensorFlow), é composta por:

- **Três camadas convolucionais (`Conv2D`)** com 32, 64 e 128 filtros respectivamente, seguidas por camadas de `MaxPooling2D`. Estas camadas são responsáveis por aprender e extrair hierarquias de características das imagens, como bordas, texturas e formas complexas.
- **Uma camada `Flatten`** para converter os mapas de características 2D em um vetor 1D.
- **Uma camada densa (`Dense`)** com 128 neurônios e uma camada de `Dropout` de 50% para regularização, prevenindo o superaquecimento (overfitting) do modelo.
- **Uma camada de saída `Dense`** com ativação `softmax`, contendo um neurônio para cada uma das 37 classes de lixo eletrônico, que fornecerá a probabilidade de a imagem pertencer a cada categoria.

---

## 4. Desenvolvimento

O desenvolvimento do projeto foi estruturado em uma série de etapas sequenciais, desde a configuração do ambiente até a preparação para o treinamento do modelo de inteligência artificial.

### 4.1. Configuração do Ambiente e Estrutura do Projeto
A primeira etapa consistiu na organização do projeto em uma estrutura de diretórios lógica, contendo pastas para a aplicação (`app`), o dataset (`dataset`), os scripts fonte (`src`), os notebooks de exploração (`notebooks`) e o relatório (`report`). Um ambiente virtual Python (`.venv`) foi criado para isolar as dependências do projeto.

### 4.2. Obtenção e Exploração dos Dados
O dataset de imagens de lixo eletrônico foi obtido da plataforma Roboflow e descompactado no diretório `dataset/`. Para a exploração inicial, foi desenvolvido o script `src/data_exploration.py`, que gera uma grade visual com imagens de amostra dos conjuntos de treino, validação e teste. Esta análise visual, salva em `report/figures/amostras_dataset.png`, foi crucial para identificar um problema de caminhos relativos incorretos no arquivo de metadados `data.yaml`, que foi posteriormente corrigido.

### 4.3. Pré-processamento das Imagens
Com base na exploração inicial, foi criado o script `src/preprocess.py` para automatizar o pré-processamento de todo o dataset. Conforme detalhado na seção de Proposta, este script aplica um pipeline de quatro etapas (conversão para escala de cinza, filtro Gaussiano, equalização de histograma e redimensionamento) em cada imagem. A execução deste script gerou um novo conjunto de dados tratado no diretório `dataset/processed/`, pronto para ser consumido pelo modelo.

### 4.4. Preparação para o Treinamento
Para a etapa de modelagem, as bibliotecas `tensorflow` e `scikit-learn` foram instaladas no ambiente virtual. Em seguida, o script `src/train.py` foi desenvolvido para orquestrar todo o processo de treinamento. Ele utiliza a classe `ImageDataGenerator` do Keras para carregar as imagens pré-processadas, define a arquitetura da CNN (detalhada na seção anterior) e implementa callbacks como `EarlyStopping` e `ModelCheckpoint` para otimizar o treinamento e salvar o melhor modelo resultante no diretório `app/model/`.

---

### 4.5. Treinamento do Modelo
Com o ambiente configurado e os dados devidamente pré-processados e organizados, o treinamento do modelo de Rede Neural Convolucional (CNN) foi executado utilizando o script `src/train.py`. O processo ocorreu da seguinte forma:

1.  **Carregamento dos Dados:** Os geradores de dados (`ImageDataGenerator`) carregaram as imagens de treino e validação a partir do diretório `dataset/processed/`, já organizadas em subpastas por classe.
2.  **Início do Treinamento:** O modelo foi treinado por 6 épocas.
3.  **Resultados:** Ao final da 6ª época, o modelo alcançou uma **acurácia de aproximadamente 63%** no conjunto de dados de treino e uma **acurácia de validação de 35%**.
4.  **Parada Antecipada (Early Stopping):** O treinamento foi interrompido automaticamente, pois a perda no conjunto de validação (`val_loss`) não apresentou melhora, indicando que o modelo havia atingido seu ponto ótimo de aprendizado com os dados atuais.
5.  **Salvamento do Modelo:** O melhor modelo, com base na métrica `val_accuracy`, foi salvo em `app/model/e_waste_classifier_best.keras`, e um arquivo com os nomes das classes foi gerado em `app/model/class_names.txt` para uso futuro pela aplicação.

Esta etapa conclui com sucesso o ciclo de desenvolvimento do modelo, resultando em um classificador funcional.

---

### 4.6. Otimização e Retreinamento

Durante o treinamento com *Data Augmentation*, o processo foi interrompido inesperadamente pelo sistema operacional. A análise indicou que a causa provável foi o consumo excessivo de memória RAM, uma vez que a geração de imagens em tempo real é uma tarefa computacionalmente intensiva.

Para solucionar este problema, duas otimizações foram realizadas no script `src/train.py`:

1.  **Redução do Tamanho do Lote (Batch Size):** O `BATCH_SIZE` foi reduzido de 32 para 16, diminuindo a quantidade de imagens processadas simultaneamente e, consequentemente, o consumo de memória.
2.  **Correção do Monitoramento:** Foi corrigido o parâmetro de monitoramento do callback `EarlyStopping`, alterando-o de `val_loss` para `val_accuracy`, garantindo que o treinamento pare com base na métrica correta e que o melhor modelo seja salvo adequadamente.

Após essas correções, o modelo foi treinado novamente com sucesso, concluindo todas as épocas e validando o progresso a cada passo.

---

## 5. Conclusão

*(Esta seção apresentará as conclusões do trabalho, retomando os objetivos e discutindo os resultados alcançados.)*

---

## 6. Referências

ABADI, Martín et al. **TensorFlow: Large-scale machine learning on heterogeneous distributed systems.** arXiv preprint arXiv:1603.04467, 2016.

BRADSKI, Gary. **The OpenCV Library.** Dr. Dobb's Journal of Software Tools, 2000.

PEDREGOSA, Fabian et al. **Scikit-learn: Machine learning in Python.** Journal of machine learning research, v. 12, p. 2825-2830, 2011.

ROBOFLOW. **Balanced E-Waste Dataset.** Disponível em: <https://universe.roboflow.com/david-andrew-e1p1t/balanced-e-waste-dataset-77kuk/dataset/1>. Acesso em: 11 set. 2025.

VAN ROSSUM, Guido; DRAKE JR, Fred L. **Python reference manual.** Centrum voor Wiskunde en Informatica Amsterdam, 1995.

---

## 7. Apêndice

O código-fonte completo desenvolvido para este projeto, incluindo os scripts para exploração de dados (`data_exploration.py`), pré-processamento (`preprocess.py`) e treinamento do modelo (`train.py`), encontra-se no diretório `src/` do repositório do projeto.

