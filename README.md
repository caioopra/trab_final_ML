# Implementação de Rede Neural Artificial (RNA)

## Integrantes do Projeto

- Caio Prá Silva
- Jose Daniel Alves do Prado
- Gustavo Konescki Fuhr

## Descrição do Projeto

O objetivo do projeto é implementar uma rede neural artificial (RNA) utilizando conceitos teóricos explorados na disciplina e recursos de baixo nível da linguagem de programação Python. A implementação segue os princípios matemáticos de funcionamento de uma rede neural, incluindo sua estrutura e métodos de treinamento. A interface utilizada para os métodos e classes se baseia na interface da biblioteca [Pytorch](https://pytorch.org).

## Estrutura do Projeto

```plaintext
Projeto de Machine Learning
├── data/                                               # Arquivos .csv com datasets
│   ├── diabetes_012_health_indicators_BRFSS2015.csv    # Dados sobre diabetes e indicadores de saúde (BRFSS 2015)
│   ├── gym_members_exercise_tracking.csv               # Dados sobre acompanhamento de exercícios dos membros da academia
│   └── wisconsin_breast_cancer.csv                     # Dados sobre câncer de mama de Wisconsin
├── README.md                                           # Documento com informações sobre o projeto
├── requirements.txt                                    # Bibliotecas necessárias para execução do projeto
└── src/                                                # Código-fonte
    ├── binary_classification.ipynb                     # Modelo de classificação binária (utilizando wisconsin_breast_cancer.csv)
    ├── loss.py                                         # Implementação das funções de perda 
    ├── metrics.py                                      # Implementação das métricas de avaliação do modelo (MSE, RMSE e SSE)
    ├── models.py                                       # Implementação dos modelos de aprendizado de máquina (definição da classe MLP e pequeno exemplo de uso)
    ├── multi_classification1.ipynb                     # Modelo de classificação multiclasse para dataset iris
    ├── multi_classification2.ipynb                     # Modelo de classificação multiclasse para dados de diabetes
    ├── nn.py                                           # Implementação de redes neurais (principais componentes de redes neurais (Neuron e Layer), bem como funções de ativação)
    ├── regression_gym_members.ipynb                    # Modelo de regressão no dataset de membros da academia, prevendo quantidade de calorias queimadas durante exercício
    └── Value.py                                        # Componente que encapsula comportamento de backpropagation e salva dados de cada valor no modelo
```


## Instruções de Uso

### Requisitos

- Python 3.10 ou superior
- Bibliotecas listadas no arquivo [requirements.txt](requirements.txt), que são bibliotecas usadas para funções não essenciais, como divisão do conjunto de teste, cálculo de precisão e outras métricas de avaliação.



### Instalação

Instale as dependências:
```sh
pip install -r requirements.txt
```

### Estrutura da Rede

#### [Value](src/Value.py)

A classe **Value** é uma das principais componentes desse projeto. Por fins didáticos e de visualização, escolheu-se adotar uma estrutura baseada em Orientação a Objetos para representação das redes neurais, ainda que com desempenho sub-ótimo. Dessa forma, a classe é responsável por armazenar um valor e eventualmente o gradiente do mesmo, que pode ser visto como um nó em um grafo. Objetos dessa classe possuem como método uma função *_backward*, que é responsável por fazer a atualização do gradiente. Para a atualização do gradiente, um nó final (no caso das redes neurais, o nó/Value que representa uma função de *loss*) deve invocar o método *.backward()*, que irá construir, de forma recursiva, a estrutura atual dos nós e atualizar o valor do gradiente.

#### [nn](src/nn.py)

Nesse arquivo definem-se os principais componentes necessários para a criação de modelos de redes neurais:

- **Module**: classe abstrata que define a estrutura básica que todos modelos devem implementar, sendo eles:
    - *parameters()*: retorna a lista de *Value*s que servem de parâmetro para a instância (como os pesos dos neurônios).
    - *\_\_call\_\_*: método "dunder" ou "magic method" do Python que é invocado ao usar uma instância de objeto "como uma função", ou seja, ao realizar `<variavel>(<parametros>)`, que é usado para passar dados de treinamento para a rede, similarmente à interface do PyTorch.
- **Neuron**: possui pesos e um viés, iniciados aleatoriamente no intervalo *[-1, 1]*; possui também uma função de ativação, usada ao final do cálculo da saída.
- **Layer**: encapsulamente de um conjunto de neurônios, que não estão interconectados.

Também são definidas as funções de ativação ReLU, Sigmoid, Softmax e tanh.

#### [models](src/models.py)

Define a classe de **Multilayer Perceptron** (MLP), que é uma arquitetura de rede neural feedforward. Pode ser definida para modelos de regressão ou classificação, recebendo como parâmetros a *quantidade de variáveis de entrada*, formando assim a primeira camada de neurônios, e *estrutura de camadas internas e de saída* (dentro de um vetor, de forma que [16, 8, 1] representa uma rede neural com 2 camadas internas, com 16 e 8 neurônios respectivamente, e 1 neurônio na camada de saída).

Nesse arquivo, também está disponível um exemplo de fluxo de treinamento e teste para redes neurais conforme a arquitetura desenvolvida. Brevemente:

- **Loop de treinamento**: é responsável por ajustar os pesos da rede neural para minimizar a função de perda. A cada época, o modelo faz previsões para os dados de treinamento, calcula a perda, realiza a retropropagação para calcular os gradientes e atualiza os pesos usando o gradiente descendente. O processo é repetido por um número definido de épocas.
    - previsões do modelo para o conjunto de treinamento;
    - cálculo da função de perda;
    - zerar os gradientes;
    - algoritmo de *backpropagation*:
        - cálculo do gradiente para todos nós da rede;
        - atualização do valor do nó, baseado no gradiente e *learning rate*;
- **Loop de teste**: é utilizado para avaliar o desempenho do modelo nos dados de teste. Após o treinamento, o modelo faz previsões para os dados de teste e calcula métricas de desempenho como precisão, recall e F1 score. Além disso, a matriz de confusão pode ser gerada para analisar a performance do modelo.
    - previsão do modelo final para o conjunto de teste;
    - cálculo de métricas e plots.

### Treinamento dos Modelos

#### Regressão

O modelo de regressão foi treinado utilizando o conjunto de dados `gym_members_exercise_tracking.csv` O notebook regression_gym_members.ipynb contém todo o processo de pré-processamento, treinamento e avaliação do modelo.

#### Classificação Binária

O modelo de classificação binária foi treinado utilizando o conjunto de dados `wisconsin_breast_cancer.csv`. O notebook binary_classification.ipynb contém todo o processo de pré-processamento, treinamento e avaliação do modelo.

#### Classificação Multiclasse

O modelo de classificação multiclasse foi treinado utilizando o conjunto de dados Iris. O notebook multi_classification1.ipynb contém todo o processo de pré-processamento, treinamento e avaliação do modelo.

### Execução

Os exemplos (dentro da confição `if __name__ == "__main__"`) podem ser executados ao entrar na pasta `src` e executar o arquivo *.py* desejado.

```sh
cd /path/to/project/src
python3 <file>.py
```

Para executar os **notebooks**, abra-os e execute as células, após a instalação dos requisitos mencionados a cima.
