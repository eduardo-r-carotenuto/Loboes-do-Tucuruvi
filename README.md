# Loboes-do-Tucuruvi
Sistema de Machine Learning desenvolvido para análise e previsão do risco de inadimplência em operações de crédito. O projeto utiliza técnicas de aprendizagem de máquina supervisionada para identificar clientes com maior probabilidade de não pagamento, comparando diferentes algoritmos e considerando o impacto financeiro associado aos erros de classificação.ObjetivoO objetivo principal é prever a probabilidade de inadimplência de um cliente a partir de dados financeiros e cadastrais, alinhando o desempenho estatístico às métricas do negócio.A tomada de decisão considera o custo assimétrico das classificações incorretas:Falso Negativo (Aprovar cliente inadimplente): prejuízo estimado de R$ 8.000,00.Falso Positivo (Recusar cliente adimplente): custo de oportunidade estimado em R$ 1.500,00.O limiar de decisão (decision threshold) é otimizado para minimizar o custo financeiro total, em vez de depender unicamente de métricas tradicionais de classificação.DadosO modelo foi desenvolvido com base num conjunto de dados sintéticos contendo 6.000 contratos de crédito.Atributos do modeloIdadeRenda mensalTempo de empregoScore de créditoNúmero de dívidas ativasPosse de imóvelFinalidade do empréstimoValor do empréstimoPrazo do empréstimoVariável Alvo (inadimplente)0: Cliente adimplente1: Cliente inadimplenteNota: O identificador id_contrato foi excluído do processo de treino para evitar memorização indesejada (overfitting).Análise Exploratória dos DadosA análise exploratória avaliou o comportamento e a distribuição das variáveis antes da fase de treino:Proporção e desbalanceamento entre classes de adimplência.Relação entre o score de crédito e a taxa de inadimplência.Distribuição do risco por finalidade do empréstimo e posse de imóvel.Identificação de dados ausentes.Os gráficos e relatórios gerados nesta etapa encontram-se salvos no diretório outputs/.Pré-processamentoO pré-processamento das variáveis é estruturado por meio das ferramentas Pipeline e ColumnTransformer da biblioteca Scikit-Learn:Variáveis Numéricas: imputação de valores ausentes pela mediana e padronização com StandardScaler.Variáveis Categóricas: imputação pela categoria mais frequente e codificação com OneHotEncoder.O encadeamento via pipeline impede o vazamento de dados (data leakage), garantindo que as transformações do conjunto de validação/teste sejam aprendidas exclusivamente no conjunto de treino.Modelos AvaliadosForam comparados os seguintes algoritmos de classificação:Regressão LogísticaRandom ForestGradient BoostingA seleção do modelo foi realizada via validação cruzada (5-fold cross-validation), tendo a métrica ROC AUC como critério principal. O conjunto de teste permaneceu isolado durante esta etapa.Resultados da Validação CruzadaModeloROC AUC MédioRegressão LogísticaA preencherRandom ForestA preencherGradient BoostingA preencherModelo selecionado: [A preencher após a execução]Avaliação do ModeloApós a escolha do algoritmo com melhor desempenho na validação cruzada, o modelo final foi avaliado sobre o conjunto de teste através das seguintes métricas:AcuráciaPrecisãoRecallF1-scoreROC AUCMatriz de ConfusãoDesempenho no Conjunto de TesteMétricaResultadoAcuráciaA preencherPrecisãoA preencherRecallA preencherF1-ScoreA preencherROC AUCA preencherA curva ROC gerada no teste é salva automaticamente em outputs/05_curva_roc.png.Escolha do Limiar de DecisãoA adoção do ponto de corte padrão (0.5) raramente reflete a melhor estratégia operacional quando os erros possuem custos distintos.A definição do limiar ótimo é feita exclusivamente sobre o conjunto de treino via validação cruzada, aplicando a seguinte função de custo:$$\text{Custo Total} = (\text{Inadimplentes Aprovados} \times 8000) + (\text{Bons Clientes Recusados} \times 1500)$$Limiar recomendado: [A preencher após a execução]Aplicação WebA solução inclui uma interface web desenvolvida em Flask para inferência em tempo real. A aplicação consome os dados do contrato fornecidos pelo utilizador e retorna:Probabilidade estimada de inadimplência.Classificação de risco (Baixo Risco / Alto Risco).Ponto de corte (limiar) utilizado.(Inserir capturas de ecrã da aplicação nos casos de baixo e alto risco no repositório)Estrutura do ProjetoPlaintextrisco-certo-ml/
│
├── data/
│   └── credito.csv
├── models/
│   ├── credito.joblib
│   └── metricas.json
├── outputs/
│   ├── 01_proporcao_inadimplencia.png
│   ├── 02_inadimplencia_score.png
│   ├── 03_inadimplencia_finalidade.png
│   ├── 04_inadimplencia_imovel.png
│   └── 05_curva_roc.png
├── templates/
│   └── index.html
├── analise_credito.py
├── app.py
├── config.py
├── gerar_dados.py
├── treinar.py
├── requirements.txt
└── README.md
Como ExecutarClonar o repositório:Bashgit clone <LINK_DO_REPOSITORIO>
cd risco-certo-ml
Criar e ativar o ambiente virtual:Bashpython -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
Instalar as dependências:Bashpip install -r requirements.txt
Gerar o conjunto de dados:Bashpython gerar_dados.py
Executar a análise exploratória:Bashpython analise_credito.py
Treinar e avaliar os modelos:Bashpython treinar.py
Iniciar a aplicação web:Bashpython app.py
Aceder ao endereço http://localhost:5000 no navegador.Governação e Uso ResponsávelSistemas preditivos aplicados à concessão de crédito exigem supervisão contínua para mitigar viés estrutural ou discriminação indireta (proxy bias). Atributos sensíveis como género, raça ou orientação religiosa não integram a modelagem.O modelo atua estritamente como uma ferramenta de auxílio à tomada de decisão (human-in-the-loop), devendo os casos limítrofes ser encaminhados para análise humana. O tratamento dos dados cumpre os requisitos de finalidade, necessidade e transparência previstos na LGPD.Tecnologias UtilizadasPythonpandasNumPyScikit-LearnMatplotlibFlaskJoblibHTML / CSS / JavaScript
