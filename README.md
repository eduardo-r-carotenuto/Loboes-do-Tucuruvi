Eduardo Dias Carotenuto 3064556/
Matheus Diório 2743334/
Daniel Gomes Santiago 2681939/
Fauzer Ribeiro da Silva 2768291/
Everton Tiburcio 2775959
# Loboes-do-Tucuruvi
Sistema de Machine Learning desenvolvido para análise e previsão do risco de inadimplência em operações de crédito.

O projeto utiliza algoritmos de aprendizado de máquina supervisionado para identificar clientes com maior probabilidade de inadimplência. Também são comparados diferentes modelos de classificação, levando em consideração não apenas o desempenho estatístico, mas também o impacto financeiro dos erros cometidos pelo modelo.

## Objetivo

O objetivo do projeto é desenvolver um modelo capaz de estimar a probabilidade de inadimplência de um cliente com base em informações financeiras e cadastrais.

Além da previsão, o projeto considera o impacto financeiro de cada decisão. Foram definidos os seguintes custos:

- Aprovar um cliente que posteriormente se torna inadimplente: prejuízo estimado de R$ 8.000.
- Recusar um cliente que pagaria corretamente: perda estimada de R$ 1.500.

Por isso, o projeto não utiliza apenas o limiar padrão de classificação de 0,5. Diferentes limiares são testados para encontrar uma configuração que reduza o custo financeiro dos erros.

## Dados utilizados

O conjunto de dados possui 6.000 contratos de crédito sintéticos. Os dados foram gerados para o projeto, portanto não são utilizados dados pessoais reais.

As principais variáveis utilizadas são:

- idade;
- renda mensal;
- tempo de emprego;
- score de crédito;
- número de dívidas ativas;
- posse de imóvel;
- finalidade do empréstimo;
- valor do empréstimo;
- prazo do empréstimo.

A variável utilizada como alvo é `inadimplente`:

- `0` — cliente adimplente;
- `1` — cliente inadimplente.

A coluna `id_contrato` não é utilizada no treinamento, pois serve apenas como identificador do contrato. Utilizá-la como variável poderia fazer o modelo aprender padrões relacionados ao identificador em vez de características relevantes para novos contratos.

## Análise exploratória

Antes do treinamento dos modelos, foi realizada uma análise dos dados para entender melhor o comportamento da base.

Foram analisados:

- proporção entre clientes adimplentes e inadimplentes;
- relação entre score de crédito e inadimplência;
- taxa de inadimplência por finalidade do empréstimo;
- relação entre posse de imóvel e inadimplência;
- existência de valores ausentes.

Os gráficos gerados nessa etapa ficam armazenados na pasta `outputs/`.

## Pré-processamento

O pré-processamento foi feito utilizando `Pipeline` e `ColumnTransformer` do Scikit-learn.

Para as variáveis numéricas:

- valores ausentes são preenchidos utilizando a mediana;
- os dados são padronizados com `StandardScaler`.

Para as variáveis categóricas:

- valores ausentes são preenchidos utilizando a categoria mais frequente;
- os dados são transformados utilizando `OneHotEncoder`.

A utilização de um pipeline ajuda a manter o processo organizado e evita que informações do conjunto de teste sejam utilizadas durante o treinamento.

## Modelos avaliados

Foram testados três algoritmos de classificação:

- Regressão Logística;
- Random Forest;
- Gradient Boosting.

A comparação foi feita utilizando validação cruzada com 5 folds. A principal métrica utilizada nessa etapa foi a ROC AUC.

O conjunto de teste foi mantido separado e não foi utilizado para escolher o modelo.

### Resultado da validação cruzada

| Modelo | ROC AUC médio |
|---|---:|
| Regressão Logística | A preencher |
| Random Forest | A preencher |
| Gradient Boosting | A preencher |

**Modelo escolhido:** A preencher após a execução.

## Avaliação do modelo

Depois da comparação, o melhor modelo é treinado utilizando os dados de treinamento e posteriormente avaliado no conjunto de teste.

As métricas utilizadas são:

- Acurácia;
- Precisão;
- Recall;
- F1-score;
- ROC AUC;
- Matriz de confusão.

### Resultados no conjunto de teste

| Métrica | Resultado |
|---|---:|
| Acurácia | A preencher |
| Precisão | A preencher |
| Recall | A preencher |
| F1-score | A preencher |
| ROC AUC | A preencher |

A curva ROC também é gerada durante a execução e salva em:

```text
outputs/05_curva_roc.png
```

## Escolha do limiar de decisão

Em problemas de classificação, é comum utilizar `0,5` como limiar para definir a classe prevista. Neste projeto, porém, esse valor pode não ser o mais adequado devido ao custo diferente de cada tipo de erro.

São considerados dois erros principais.

### Falso positivo

O modelo classifica um cliente adimplente como cliente de alto risco.

Nesse caso, um cliente que provavelmente pagaria o empréstimo é recusado.

**Custo estimado: R$ 1.500.**

### Falso negativo

O modelo classifica um cliente inadimplente como cliente de baixo risco e o empréstimo é aprovado.

**Custo estimado: R$ 8.000.**

Como o falso negativo possui um custo maior, são avaliados diferentes limiares de classificação.

O custo total é calculado da seguinte forma:

```text
Custo total =
(inadimplentes aprovados × R$ 8.000)
+
(bons clientes recusados × R$ 1.500)
```

O limiar escolhido é aquele que apresenta o menor custo financeiro estimado.

A definição do limiar utiliza apenas informações provenientes do treinamento, por meio de validação cruzada. O conjunto de teste não é utilizado para tomar essa decisão.

**Limiar escolhido:** A preencher após a execução.

Dessa forma, a escolha do modelo não fica baseada somente em métricas estatísticas, mas também considera o impacto que cada tipo de erro pode causar no problema analisado.

## Aplicação Web

O modelo treinado foi integrado a uma aplicação web desenvolvida com Flask.

Na aplicação, o usuário pode informar os dados de um novo contrato e consultar:

- probabilidade estimada de inadimplência;
- classificação de risco;
- limiar utilizado para a decisão.

### Exemplo de cliente de baixo risco

Adicionar captura de tela da aplicação.

### Exemplo de cliente de alto risco

Adicionar captura de tela da aplicação.

## Estrutura do projeto

```text
risco-certo-ml/
│
├── data/
│   └── credito.csv
│
├── models/
│   ├── credito.joblib
│   └── metricas.json
│
├── outputs/
│   ├── 01_proporcao_inadimplencia.png
│   ├── 02_inadimplencia_score.png
│   ├── 03_inadimplencia_finalidade.png
│   ├── 04_inadimplencia_imovel.png
│   └── 05_curva_roc.png
│
├── templates/
│   └── index.html
│
├── analise_credito.py
├── app.py
├── config.py
├── gerar_dados.py
├── treinar.py
├── requirements.txt
└── README.md
```

## Como executar

### 1. Clone o repositório

```bash
git clone LINK_DO_REPOSITORIO
cd risco-certo-ml
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Gere os dados

```bash
python gerar_dados.py
```

Esse comando cria o conjunto de dados utilizado no projeto.

### 5. Execute a análise exploratória

```bash
python analise_credito.py
```

Os gráficos gerados serão armazenados na pasta `outputs/`.

### 6. Treine e avalie os modelos

```bash
python treinar.py
```

Esse comando realiza a validação cruzada, compara os modelos, seleciona o melhor algoritmo, avalia os diferentes limiares e realiza a avaliação final no conjunto de teste.

### 7. Inicie a aplicação

```bash
python app.py
```

Depois, acesse:

```text
http://localhost:5000
```

## Ética e uso responsável

Modelos utilizados em decisões de crédito podem reproduzir padrões existentes nos dados utilizados para treinamento. Por isso, os resultados devem ser analisados com cuidado, principalmente em situações que podem afetar diretamente a vida financeira de uma pessoa.

Características sensíveis, como raça, gênero, religião e orientação sexual, não devem ser utilizadas como critérios para concessão de crédito. Também é importante verificar se outras variáveis utilizadas pelo modelo não acabam funcionando como substitutas indiretas dessas características.

Por se tratar de uma decisão de alto impacto, o modelo deve ser considerado uma ferramenta de apoio à decisão, e não necessariamente o único responsável pela aprovação ou recusa de um cliente.

Também é importante permitir revisão humana em situações específicas, principalmente quando uma decisão automatizada puder gerar consequências relevantes.

Em relação à LGPD, o tratamento de dados deve ter uma finalidade definida, utilizar apenas as informações necessárias, garantir segurança e transparência e respeitar os direitos dos titulares, inclusive nos casos de decisões automatizadas quando aplicável.

## Tecnologias utilizadas

- Python
- pandas
- NumPy
- Scikit-learn
- Matplotlib
- Flask
- Joblib
- HTML
- CSS
- JavaScript

## Autora

**Mylena Martins dos Santos**

Projeto desenvolvido para a disciplina de Inteligência Artificial e Machine Learning.
