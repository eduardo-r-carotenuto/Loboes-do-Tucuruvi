# Loboes-do-Tucuruvi
Sistema de Machine Learning desenvolvido para análise e previsão do risco de inadimplência em operações de crédito.

O projeto utiliza técnicas de aprendizado de máquina supervisionado para identificar clientes com maior probabilidade de inadimplência, comparando diferentes algoritmos e considerando também o impacto financeiro dos erros de classificação.

---

## 🎯 Objetivo

O objetivo do projeto é desenvolver um modelo capaz de prever a probabilidade de inadimplência de um cliente a partir de informações financeiras e cadastrais.

Além da capacidade preditiva, o projeto considera o impacto financeiro das decisões:

- Aprovar um cliente que se torna inadimplente gera um prejuízo estimado de **R$ 8.000**.
- Recusar um cliente que pagaria corretamente representa uma perda estimada de **R$ 1.500**.

Por esse motivo, a escolha do limiar de classificação não considera apenas métricas tradicionais, mas também o custo associado a cada tipo de erro.

---

## 📊 Dados

O conjunto de dados utilizado contém **6.000 contratos de crédito sintéticos**, permitindo desenvolver e avaliar os modelos sem utilizar dados pessoais reais.

As variáveis analisadas incluem:

- idade;
- renda mensal;
- tempo de emprego;
- score de crédito;
- número de dívidas ativas;
- posse de imóvel;
- finalidade do empréstimo;
- valor do empréstimo;
- prazo do empréstimo.

A variável alvo é `inadimplente`:

- `0` → cliente adimplente;
- `1` → cliente inadimplente.

A coluna `id_contrato` não foi utilizada como feature, pois funciona apenas como identificador e poderia permitir que o modelo memorizasse exemplos sem aprender padrões úteis para novos contratos.

---

## 🔎 Análise Exploratória dos Dados

Antes do treinamento dos modelos foi realizada uma análise exploratória para compreender o comportamento dos dados.

Foram analisados:

- proporção entre clientes adimplentes e inadimplentes;
- relação entre score de crédito e inadimplência;
- taxa de inadimplência por finalidade do empréstimo;
- relação entre posse de imóvel e inadimplência;
- presença de valores ausentes.

Os gráficos produzidos durante essa etapa são armazenados na pasta `outputs/`.

---

## 🧹 Pré-processamento

O pré-processamento foi realizado utilizando `Pipeline` e `ColumnTransformer` do Scikit-learn.

Para as variáveis numéricas foram aplicados:

- preenchimento de valores ausentes pela mediana;
- padronização com `StandardScaler`.

Para as variáveis categóricas foram aplicados:

- preenchimento pela categoria mais frequente;
- transformação por `OneHotEncoder`.

O uso do pipeline ajuda a evitar **data leakage**, garantindo que as transformações sejam aprendidas apenas a partir dos dados de treinamento.

---

## 🤖 Modelos avaliados

Foram comparados três algoritmos de classificação:

1. Regressão Logística
2. Random Forest
3. Gradient Boosting

A comparação foi realizada utilizando **validação cruzada com 5 folds**, considerando como principal métrica a **ROC AUC**.

O conjunto de teste foi mantido separado e não foi utilizado para selecionar o algoritmo.

### Resultado da validação cruzada

| Modelo | ROC AUC médio |
|---|---:|
| Regressão Logística | A preencher |
| Random Forest | A preencher |
| Gradient Boosting | A preencher |

**Modelo escolhido:** A preencher após a execução.

---

## 📈 Avaliação do modelo

Após a seleção do melhor algoritmo, o modelo escolhido foi treinado utilizando o conjunto de treinamento e avaliado no conjunto de teste.

Foram consideradas as seguintes métricas:

- Acurácia
- Precisão
- Recall
- F1-score
- ROC AUC
- Matriz de confusão

### Resultados no conjunto de teste

| Métrica | Resultado |
|---|---:|
| Acurácia | A preencher |
| Precisão | A preencher |
| Recall | A preencher |
| F1-score | A preencher |
| ROC AUC | A preencher |

A curva ROC também é gerada automaticamente durante a execução e armazenada em:

`outputs/05_curva_roc.png`

---

## 💰 Escolha do limiar de decisão

O limiar padrão utilizado em muitos problemas de classificação é `0.5`. Entretanto, esse valor não necessariamente representa a melhor decisão para o negócio.

Neste projeto foram considerados dois tipos principais de erro.

### Falso positivo

O modelo classifica um cliente bom como alto risco.

Nesse caso, um cliente que provavelmente pagaria corretamente o empréstimo é recusado.

**Custo estimado: R$ 1.500.**

### Falso negativo

O modelo classifica um cliente inadimplente como baixo risco e aprova o empréstimo.

**Custo estimado: R$ 8.000.**

Como o falso negativo apresenta impacto financeiro maior, diferentes limiares são avaliados.

A escolha do limiar é realizada utilizando previsões obtidas somente a partir dos dados de treinamento por meio de validação cruzada, sem utilizar o conjunto de teste para tomar essa decisão.

Para cada limiar é calculado o custo:

`Custo total = (inadimplentes aprovados × R$ 8.000) + (bons clientes recusados × R$ 1.500)`

O limiar recomendado é aquele que apresenta o menor custo financeiro estimado.

**Limiar escolhido:** A preencher após a execução.

Essa estratégia permite transformar a previsão estatística do modelo em uma decisão mais alinhada ao problema de negócio.

---

## 🌐 Aplicação Web

O modelo treinado foi integrado a uma aplicação desenvolvida com **Flask**.

O usuário pode informar os dados de um novo contrato e receber:

- probabilidade estimada de inadimplência;
- classificação de baixo ou alto risco;
- limiar utilizado para a decisão.

### Exemplo de cliente de baixo risco

Adicionar captura de tela da aplicação.

### Exemplo de cliente de alto risco

Adicionar captura de tela da aplicação.

---

## 📁 Estrutura do projeto

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

---

## 🚀 Como executar

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

Esse comando cria o conjunto de dados utilizado pelo projeto.

### 5. Execute a análise exploratória

```bash
python analise_credito.py
```

Os gráficos serão armazenados na pasta `outputs/`.

### 6. Treine e avalie os modelos

```bash
python treinar.py
```

O programa realiza a validação cruzada, compara os algoritmos, seleciona o melhor modelo, avalia diferentes limiares e realiza a avaliação final.

### 7. Inicie a aplicação

```bash
python app.py
```

Depois, acesse no navegador:

```text
http://localhost:5000
```

---

## ⚖️ Ética e uso responsável

Modelos utilizados em decisões de crédito podem reproduzir desigualdades existentes nos dados históricos e transformar padrões sociais em decisões aparentemente neutras.

Mesmo que aumentassem o desempenho do modelo, atributos sensíveis como raça, gênero, religião e orientação sexual não deveriam ser utilizados como critérios para concessão de crédito.

Também é importante avaliar variáveis que possam funcionar indiretamente como proxies para características sensíveis.

Por se tratar de uma decisão de alto impacto, o modelo deve ser utilizado como uma ferramenta de **apoio à decisão**, e não necessariamente como único responsável pela aprovação ou recusa de um cliente.

Casos específicos devem permitir revisão humana, principalmente quando a decisão pode gerar consequências relevantes para o indivíduo.

Em relação à **LGPD**, o tratamento de dados deve possuir finalidade definida, utilizar apenas as informações necessárias, garantir segurança e transparência e respeitar os direitos dos titulares, inclusive em decisões automatizadas quando aplicável.

---

## 🛠️ Tecnologias utilizadas

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

---

## 👩💻 Autora

**Mylena Martins dos Santos**

Projeto desenvolvido para a disciplina de **Inteligência Artificial e Machine Learning**.
