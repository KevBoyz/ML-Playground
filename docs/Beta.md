# Plano Beta — Orquestração de projetos tabulares de ML

## 1. Objetivo

Evoluir o ML-Playground de um executor de experimentos de classificação para
uma plataforma capaz de reproduzir os fluxos recorrentes dos notebooks de
`notebooks/` por configuração. Um novo projeto deverá ser criado adicionando
um dataset adequado e uma pasta de experimento; o usuário não deverá escrever
um notebook gigante nem código de cola para carregar dados, comparar modelos,
avaliar resultados e publicar artefatos.

O Beta deve atender três famílias de tarefas:

- **classificação** binária e multiclasses;
- **regressão** com alvo numérico;
- **clusterização/segmentação** sem alvo supervisionado.

Análise exploratória e estatística será uma capacidade transversal e, quando
necessário, uma extensão posterior. Ela não deve bloquear os três fluxos
principais.

### Premissa de entrada

O pacote receberá um dataset tabular relativamente adequado para o algoritmo:
colunas legíveis, tipos coerentes, target definido quando aplicável e uma
seleção de atributos razoável. O pacote não tentará transformar um dataset
bruto de domínio específico em dataset de modelagem.

O pacote deve orquestrar:

1. carregamento e validação do contrato do dataset;
2. expansão de candidatos e hiperparâmetros;
3. treinamento, validação e comparação;
4. seleção do resultado vencedor;
5. persistência do artefato, predições, métricas e figuras;
6. manifesto, logs e resumo reproduzível da execução.

## 2. Evidências extraídas dos notebooks

Foram observados 15 notebooks. Apesar das diferenças de domínio, os mesmos
passos aparecem repetidamente:

| Família | Evidências | Padrão que deve virar coordenação |
| --- | --- | --- |
| Classificação | `03- Apostas_futebol`, `04-Titanic`, `08-Wine`, `12-heartdiseaseKNN`, `14-heartchains` e a etapa supervisionada de `13-credit-card-segmentation` | comparar vários estimadores, usar holdout/CV/grid, calcular métricas por classe, inspecionar matriz de confusão e salvar o melhor pipeline |
| Regressão | `01VendasShopping`, `07-Salary_prediction`, `10-life_expectation`, `11-crazy_reg` | comparar regressão linear e ensembles, executar K-Fold/Grid Search, medir MAE/MSE/RMSE/R², analisar resíduos e persistir predições |
| Clusterização | `05-Customer_segmentation` e `13-credit-card-segmentation` | testar K-Means, DBSCAN e hierárquico, comparar normalizações fornecidas, medir qualidade interna, inspecionar tamanhos dos grupos e projetar em PCA |
| Diagnóstico estatístico | `02-Stat_série_A` e `09-cars` | produzir resumos, agregações, distribuições e testes configurados pelo usuário |
| Exploração e preparação | `04-Titanic`, `06-CaliforniaHouses`, `07-Salary_prediction`, `10-life_expectation`, `12-heartdiseaseKNN` e `14-heartchains` | verificar estrutura, tipos, missingness, distribuição do target, correlação e separação de treino/teste antes de modelar |

Os notebooks também mostram o que não deve ser absorvido pelo pacote: download
via Kaggle/UCI, parsing de colunas de salário e placar, criação de atributos de
negócio, tratamento manual de outliers, transformações Box-Cox/logit, seleção
manual de features, decisões de domínio e hipóteses estatísticas não
declaradas.

## 3. Princípios de escopo

### 3.1 O que entra no Beta

- uma API/CLI de execução orientada a pastas de experimento;
- runners de classificação, regressão e clusterização;
- registry de modelos com metadados de tarefa e capacidades;
- avaliação e seleção específicas por tarefa;
- pipeline opcional composto por transformadores genéricos explicitamente
  configurados pelo usuário;
- relatórios e artefatos com layout único e consciente da tarefa;
- exemplos mínimos e testes de fumaça que não dependem dos notebooks.

### 3.2 O que fica fora do núcleo

- download de datasets ou credenciais de Kaggle, UCI e serviços externos;
- execução, conversão ou importação automática de notebooks;
- parsing de strings de domínio, agregações de negócio ou criação de features;
- imputação, remoção de outliers, seleção automática de atributos ou
  transformações de distribuição como comportamento implícito;
- inferência causal, interpretação estatística automática ou conclusão sobre
  significância;
- AutoML, otimização bayesiana, deep learning, execução distribuída e séries
  temporais especializadas.

Se um usuário quiser limpeza, engenharia de atributos ou uma transformação
específica, deverá entregar o dataset preparado ou declarar explicitamente um
transformador genérico suportado. O Beta não deve esconder essas decisões em
defaults.

## 4. Estado atual e lacunas

O projeto já possui o esqueleto de um MVP de classificação: configuração em
YAML, loader de datasets, registry de estimadores, pipelines com
`ColumnTransformer`, validação cruzada, métricas, persistência e relatórios em
`reports/<experiment_name>/`. `docs/PLAN.md` continua sendo a referência para
concluir e estabilizar esse MVP.

As lacunas relevantes para o Beta são:

1. `config.py` aceita apenas `classification` e exige target para todo dataset.
2. `executor.py` assume que toda execução é supervisionada e usa métricas de
   classificação no caminho de CV.
3. O registry conhece principalmente classificadores; ainda não há um contrato
   de tarefa/capacidade para regressors e clusterizadores.
4. As métricas de regressão já existem no módulo de avaliação, mas não estão
   conectadas a um runner, à seleção ou ao relatório final.
5. Não há execução de clusterização, comparação de `n_clusters`, avaliação
   interna, tratamento de ruído ou relatório de distribuição de grupos.
6. A seleção atual pressupõe que maior é melhor; métricas como RMSE e Davies-
   Bouldin exigem direção configurável.
7. O contrato de relatórios não contempla claramente resíduos, atribuições de
   cluster, ruído, curva elbow ou perfis de grupo.
8. O pacote ainda precisa distinguir validação de modelo supervisionado de
   seleção interna de clusterização.
9. A suíte de testes não cobre um caminho completo de regressão nem de
   clusterização.

## 5. Contrato Beta de experimento

Cada projeto continuará autocontido em `experiments/<nome>/`. O formato atual
de cinco arquivos será preservado onde fizer sentido; a validação será
condicional à tarefa.

```text
experiments/
  iris_baseline/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
    views.yaml
  salary_regression/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
    views.yaml
  customer_segments/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    views.yaml
```

### 5.1 Configuração comum

`experiment.yaml` deve conter:

```yaml
name: customer_segments
task: clustering

data:
  path: data/raw/customers.csv
  features:
    - annual_income
    - spending_score
  random_state: 42

outputs:
  root: reports
  save_model: true
  save_predictions: true
  figures: true

selection:
  primary_metric: silhouette
  direction: maximize
```

Regras do contrato:

- `task` deve ser `classification`, `regression` ou `clustering`;
- classificação e regressão exigem `data.target`;
- clusterização não exige target e deve aceitar `data.features` explícito;
- quando `features` não for informado em tarefa supervisionada, o loader pode
  usar todas as colunas exceto o target, desde que registre essa decisão;
- colunas de identificação não devem ser descartadas silenciosamente: o
  usuário deve informar as features ou preparar o dataset;
- caminhos relativos devem ser resolvidos a partir da raiz do projeto;
- o loader deve registrar nomes, tipos, quantidade de linhas, features efetivas
  e uma impressão digital do dataset no manifesto;
- qualquer incompatibilidade de target, tipo ou coluna deve falhar antes do
  treinamento com o campo exato indicado na mensagem.

`preprocessing.yaml` deve representar apenas transformações genéricas
explicitamente escolhidas. O default do Beta é identidade/passthrough. O
executor não deve inventar normalização, imputação ou codificação porque um
modelo costuma se beneficiar delas.

### 5.2 Modelos e grids

`models.yaml` continua declarando candidatos e valores do grid:

```yaml
models:
  - name: random_forest_regressor
    params:
      n_estimators: [100, 300]
      max_depth: [null, 10]
      random_state: [42]
```

O loader deve validar o nome contra o registry da tarefa e rejeitar uma
combinação incompatível antes de iniciar o grid. Não deve haver importação
silenciosa de modelos opcionais que não estejam instalados.

### 5.3 Validação e seleção

- classificação: `holdout`, `kfold`, `stratified_kfold`,
  `repeated_kfold` e `repeated_stratified_kfold`;
- regressão: `holdout`, `kfold` e `repeated_kfold`, sem estratificação
  automática;
- clusterização: seleção interna no dataset fornecido; validação cruzada
  comum fica fora do Beta inicial;
- toda métrica de seleção deve declarar `maximize` ou `minimize`;
- desempates devem considerar uma métrica secundária e depois a variabilidade,
  de forma determinística;
- combinações que falharem devem ser registradas e não esconder o erro;
- a execução deve falhar apenas quando nenhuma combinação válida concluir.

## 6. Arquitetura de execução

O fluxo alvo é:

```text
CLI
  -> loader/compositor/validador
  -> leitura e contrato do dataset
  -> runner da tarefa
       classification -> avaliação supervisionada
       regression    -> avaliação supervisionada
       clustering    -> avaliação interna e atribuição de grupos
  -> seleção do melhor candidato
  -> persistência de modelos e relatórios
  -> manifesto, summary e log
```

### 6.1 Orquestrador comum

Adicionar uma camada de orquestração que escolha o runner pela tarefa e
coordene ciclo de vida, identificador de execução, tratamento de erros,
seleção e escrita. Ela deve reutilizar lógica real entre tarefas, mas não criar
funções que apenas renomeiam chamadas de `sklearn`.

Responsabilidades do orquestrador:

- normalizar a configuração;
- construir combinações de modelo e parâmetros;
- criar pipelines novos para cada fold/candidato;
- coletar resultados escalares e não escalares;
- aplicar a regra de seleção;
- encaminhar todos os artefatos ao writer central;
- produzir um resumo final com processados, concluídos, falhos e ignorados.

### 6.2 Registry orientado a capacidades

O registry deve associar cada entrada a:

- tarefa suportada;
- fábrica da classe do estimador;
- parâmetros aceitos pelo estimador;
- suporte a `predict`, `predict_proba`, `decision_function` ou `fit_predict`;
- possibilidade de importância de atributos;
- necessidade de dados numéricos;
- dependência opcional e mensagem de instalação.

Entradas prioritárias, baseadas nos notebooks:

| Tarefa | Beta P0 | Beta P1 |
| --- | --- | --- |
| Classificação | KNN, regressão logística, SVM, árvore de decisão | Random Forest, Gaussian Naive Bayes, VotingClassifier configurável |
| Regressão | DummyRegressor, LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor | GradientBoostingRegressor e outros apenas se houver demanda recorrente |
| Clusterização | KMeans | DBSCAN, AgglomerativeClustering, LocalOutlierFactor como modo de detecção de anomalias |

Pipelines compostos, como votação, só entram quando o registry conseguir
serializar a composição e reportar seus componentes. Não criar uma função
wrapper de uma única classe.

## 7. Fases de implementação

### Fase 1 — Contrato multi-tarefa e despacho (P0)

- Ampliar o schema para as três tarefas e tornar `target` condicional.
- Validar `features`, tipos necessários, target numérico na regressão e
  cardinalidade mínima para classificação/clusterização.
- Separar validação de configuração, validação de dataset e validação de
  compatibilidade modelo-métrica.
- Refatorar o executor para escolher um runner por `task`.
- Corrigir o registry para não importar dependências opcionais sem necessidade.
- Preservar o caminho atual de classificação e executar os exemplos existentes
  sem mudança de saída incompatível.
- Adicionar no manifesto o schema observado e a impressão digital do dataset.

**Aceite:** uma pasta de configuração determina a tarefa sem código Python
auxiliar; uma configuração inválida falha antes de chamar `fit`.

### Fase 2 — Runner de regressão (P0)

- Implementar o fluxo comum de holdout, K-Fold e Repeated K-Fold para alvo
  contínuo.
- Adicionar os seis modelos P0 ao registry com grids pequenos e executáveis.
- Suportar baseline `DummyRegressor` para contextualizar qualquer ganho.
- Registrar predições out-of-fold/holdout, duração por candidato e tamanhos das
  partições.
- Conectar `mae`, `mse`, `rmse`, `r2`, `mape` e `max_error` ao schema, com
  métricas escalares e direção padrão documentada.
- Gerar tabela de resíduos com observado, predito e erro quando houver
  predição salva; não fazer transformação automática do target.
- Persistir o pipeline completo do vencedor e permitir `predict` com as mesmas
  colunas declaradas no experimento.

**Aceite:** um experimento de regressão consegue comparar linear, regularizado,
floresta e baseline, selecionar por RMSE ou R² e produzir métricas por fold,
resumo, resíduos, predições e modelo.

### Fase 3 — Runner de clusterização e segmentação (P0/P1)

- Implementar `fit_predict`/`predict` para KMeans e a interface comum dos
  clusterizadores que produzem labels.
- Adicionar DBSCAN e AgglomerativeClustering sem pressupor que todos possuem
  `predict` ou `n_clusters`.
- Permitir grid de candidatos por algoritmo, incluindo varredura de `k` para
  KMeans e dos parâmetros relevantes para DBSCAN.
- Calcular silhouette, Calinski-Harabasz, Davies-Bouldin, inertia quando
  aplicável, quantidade de clusters, tamanho mínimo/máximo e proporção de
  ruído (`label == -1`).
- Marcar métricas indefinidas, como silhouette com menos de dois clusters,
  como `not_available` com motivo; nunca converter o caso em zero silencioso.
- Selecionar o vencedor somente com métricas válidas e direção compatível.
- Salvar a atribuição de cluster por linha, preservando um identificador de
  linha estável e as features usadas.
- Gerar elbow/silhouette para candidatos KMeans e projeção PCA 2D/3D apenas
  como visualização; PCA não deve entrar no treinamento sem ser configurado.
- Registrar explicitamente que clusterização não possui acurácia sem labels
  externos. Comparação supervisionada opcional fica fora do P0.

**Aceite:** o projeto de segmentação pode comparar KMeans, DBSCAN e hierárquico
por YAML, escolher por silhouette/Davies-Bouldin e publicar labels, métricas,
tamanhos de grupo e figuras sem depender de notebook.

### Fase 4 — Ampliação da classificação e diagnóstico (P1)

- Adicionar Random Forest e GaussianNB ao registry.
- Suportar métricas e relatórios por classe para problemas binários e
  multiclasses, inclusive balanced accuracy, PR AUC quando houver score e
  matriz de confusão tabular.
- Produzir curva ROC/PR, comparação de modelos, importância ou coeficientes
  somente quando o estimador expuser a capacidade correspondente.
- Adicionar curvas de aprendizado/validação como artefatos opt-in, sem executar
  múltiplos treinos inesperados no grid padrão.
- Permitir um VotingClassifier configurável somente após definir o contrato de
  componentes, parâmetros, métricas e persistência.

**Aceite:** os fluxos de Wine, Heart Disease, Titanic e futebol podem ser
  representados por dataset preparado + YAML, sem copiar a sequência manual de
  `fit`, `predict`, `GridSearchCV` e métricas.

### Fase 5 — Relatórios unificados por tarefa (P0/P1)

Manter a separação atual por experimento e acrescentar apenas artefatos que
tenham significado para a tarefa:

```text
reports/<experiment_name>/
  metrics/
    <run_id>_candidate_metrics.csv
    <run_id>_fold_metrics.csv
    <run_id>_summary.csv
  tables/
    <run_id>_model_comparison.csv
    <run_id>_manifest.json
    <run_id>_confusion_matrix.csv        # classificação
    <run_id>_residual_summary.csv        # regressão
    <run_id>_cluster_sizes.csv           # clusterização
  predictions/
    <run_id>_predictions.csv
  figures/
    <run_id>_metric_comparison.png
    <run_id>_roc_pr.png                  # classificação, se habilitado
    <run_id>_residuals.png               # regressão, se habilitado
    <run_id>_elbow_silhouette.png        # clusterização, se aplicável
    <run_id>_cluster_projection.png      # clusterização, se habilitado
```

O writer deve:

- receber resultados normalizados, não decidir a lógica do experimento;
- escrever somente em `reports/<experiment_name>/`;
- manter métricas escalares separadas de tabelas e matrizes;
- usar `run_id` em todos os nomes, sem sobrescrever execuções anteriores;
- omitir artefatos não aplicáveis ou registrar o motivo no manifesto;
- armazenar config efetiva, versões, seed, dataset, features, target, modelos,
  seleção, erros e caminhos de saída;
- produzir CSV/JSON legíveis, sem serializar objetos Python como texto opaco.

Modelos persistidos devem continuar em:

```text
models/<experiment_name>/<run_id>/model.joblib
```

O artefato deve incluir pipeline, metadata da tarefa e schema esperado. Para
clusterizadores sem `predict`, o pacote deve documentar no metadata que o
artefato serve para atribuir labels somente quando a classe suportar essa
operação.

### Fase 6 — Perfil de dataset e estatística configurável (P1/P2)

Adicionar um diagnóstico genérico e opt-in, sem transformá-lo em limpeza:

- shape, tipos, missingness, cardinalidade e duplicidade;
- resumo estatístico numérico e categórico;
- distribuição do target e proporção de classes;
- correlação configurada e gráficos básicos;
- para regressão, distribuição do target, resíduos e observado versus predito;
- para clusterização, perfil agregado por grupo e contribuição das features.

Como extensão P2, incluir uma tarefa `analysis` ou uma seção de análises que
aceite testes declarados pelo usuário, por exemplo Shapiro-Wilk ou Welch
`t-test`, com colunas, grupos, hipótese e nível de significância explícitos.
O pacote deve reportar estatística e p-valor, mas não declarar automaticamente
que uma hipótese de negócio foi provada.

**Aceite:** os notebooks estatísticos podem ser reduzidos a dados preparados e
uma configuração declarativa de análise, sem embutir regras específicas de
futebol, carros ou vendas no core.

### Fase 7 — Qualidade, logs e documentação (P0)

- Criar testes de contrato para cada tarefa, compatibilidade de modelos e
  métricas inválidas.
- Criar testes de fumaça com datasets sintéticos pequenos para classificação,
  regressão e clusterização; usar diretórios temporários para artefatos.
- Testar seleção com métricas de maximização e minimização, empate, candidato
  falho, métrica indefinida e rerun.
- Testar persistência e reabertura do modelo vencedor, incluindo schema.
- Garantir que testes não criem logs persistentes.
- Para scripts reais, obedecer `logs/<script>.log`: sobrescrever a execução,
  registrar apenas warnings/erros durante o processamento, emitir summary e
  colocar o tempo de execução na última linha.
- Atualizar `README.md` e `USAGE.md` com três exemplos mínimos: classificação,
  regressão e clusterização.
- Documentar que notebooks são referências de domínio, não dependências de
  execução.

## 8. Critérios de aceite do Beta

O Beta estará pronto quando:

- um usuário puder criar um novo projeto somente com dataset adequado e YAMLs;
- `uv run ml-playground --experiment <pasta>` despachar corretamente as três
  tarefas;
- classificação mantiver o comportamento do MVP e aceitar os modelos P1;
- regressão comparar pelo menos os modelos P0, baseline e validações definidas;
- clusterização comparar KMeans, DBSCAN e hierárquico com métricas internas;
- direção de seleção funcionar para métricas de maior e menor valor;
- cada tarefa salvar resultados, predições/labels, manifesto e artefato quando
  habilitado;
- nenhum runner fizer limpeza, parsing de domínio ou feature engineering
  implícito;
- erros de candidato aparecerem no relatório sem corromper os demais resultados;
- duas execuções e dois experimentos permaneçam isolados em `reports/` e
  `models/`;
- os testes de fumaça reproduzirem os três fluxos sem abrir notebooks;
- os logs reais obedecerem às leis do projeto.

## 9. Fora do Beta

Não bloquear este plano com:

- conversão dos 15 notebooks para módulos;
- suporte a dados não tabulares, imagens, texto ou deep learning;
- download/versionamento de datasets externos;
- limpeza automática ou catálogo de centenas de transformadores;
- otimização bayesiana, AutoML, tracking externo ou execução distribuída;
- seleção automática de atributos, target transforms ou inferência estatística
  avançada;
- monitoramento de produção, serving HTTP e atualização online.

O resultado esperado é uma camada de coordenação reproduzível para os padrões
comuns dos notebooks, mantendo as decisões específicas de cada domínio na
etapa de preparação do dataset e na configuração explícita do usuário.
 
## 10. Views diagnosticas configuraveis

Os notebooks atuais e os novos arquivos em `notebooks/more/` mostram que
metricas numericas nao bastam para entender o comportamento do modelo. Cada
experimento deve ter um `views.yaml` que selecione explicitamente as figuras a
serem geradas. O arquivo nao deve conter codigo Python nem nomes de funcoes
arbitrarias; ele seleciona renderers do registry de views e fornece parametros
validados.

Exemplo para classificacao:

```yaml
views:
  common:
    - name: model_comparison
      enabled: true
      scope: candidates
      params:
        metric: f1_macro

  classification:
    - name: confusion_matrix
      enabled: true
      scope: best
      params:
        normalize: true
        include_counts: true

    - name: roc_curve
      enabled: true
      scope: best
      params:
        multiclass: ovr

    - name: decision_boundary
      enabled: true
      scope: best
      params:
        projection: pca
        dimensions: 2

    - name: learning_curve
      enabled: false
      scope: best
      params:
        scoring: f1_macro
        train_sizes: 5
```

Regras do contrato:

- cada entrada possui `name`, `enabled`, `scope` e `params` opcionais;
- `scope` pode ser `best`, `candidates`, `selected_models` ou `folds`;
- views caras ou que treinam novamente ficam desabilitadas quando nao forem
  declaradas;
- o loader valida nome, tarefa, parametros e dependencias antes da execucao;
- uma view consome artefatos normalizados do runner, como predicoes, scores,
  probabilidades, labels, residuos, folds ou centroides;
- uma view incompativel fica `skipped`, com motivo no manifesto, sem criar uma
  figura enganosa ou interromper outros candidatos;
- views baseadas em holdout/CV usam as predicoes do split correspondente;
  nenhum renderer ajusta scaler, PCA ou modelo usando o teste somente para
  construir a figura;
- cada figura recebe nome estavel com `run_id`, candidato, fold quando
  aplicavel e nome da view;
- a configuracao permite limitar candidatos, amostra, features e pontos da
  malha, evitando grids graficos acidentalmente enormes.

#### Views supervisionadas

O catalogo consolida os graficos encontrados nos notebooks existentes e em
`notebooks/more/Supervised/`.

| View | Tarefa | Diagnostico e dependencias |
| --- | --- | --- |
| `model_comparison` | todas | barras/linhas para comparar metrica por candidato |
| `confusion_matrix` | classificacao | matriz de erro absoluta e normalizada; exige `y_true`/`y_pred` |
| `roc_curve` | classificacao | curva ROC binaria ou one-vs-rest/macro multiclasses; exige probabilidade ou score |
| `precision_recall_curve` | classificacao | comportamento em classes desbalanceadas; exige score e esquema binario/OVR |
| `class_distribution` | classificacao | quantidade e proporcao por classe antes das metricas |
| `decision_boundary` | classificacao | regiao prevista e amostras em 1D/2D; exige duas dimensoes ou `projection: pca/lda` |
| `probability_curve` | classificacao | probabilidade/score contra uma feature em modelos como regressao logistica |
| `learning_curve` | classificacao/regressao | treino versus validacao conforme tamanho da amostra; executa CV adicional somente quando habilitada |
| `validation_curve` | classificacao/regressao | score conforme hiperparametro, como `C`, `n_neighbors` ou grau |
| `knn_neighbors_curve` | classificacao | desempenho conforme `n_neighbors`, inspirado no diagnostico de Heart Disease |
| `tree_structure` | classificacao/regressao | desenho da arvore e profundidade/folhas; exige estimador de arvore |
| `feature_importance` | classificacao/regressao | coeficientes ou importancia nativa/permutacao, marcando a origem |

`decision_boundary` nao deve fingir que representa todas as features. O
manifesto informa se foram usadas duas colunas originais, PCA ou LDA, quais
dimensoes foram exibidas e qual modelo foi ajustado nesse espaco.

#### Views de regressao

Os notebooks de regressao adicionam diagnosticos que nao podem ser reduzidos a
uma unica metrica:

| View | Diagnostico |
| --- | --- |
| `predicted_vs_actual` | observado contra predito, com linha de previsao perfeita |
| `fit_vs_feature` | pontos e curva/plano ajustado quando uma ou duas features sao selecionadas explicitamente |
| `residuals_vs_fitted` | residuos contra valores ajustados, com linha zero |
| `residual_distribution` | histograma/densidade dos erros |
| `qq_residuals` | normalidade aproximada dos residuos por QQ-plot |
| `scale_location` | raiz do residuo padronizado contra ajustado para investigar variancia nao constante |
| `residuals_vs_leverage` | alavancagem, residuos padronizados e Cook quando o estimador expuser esses diagnosticos |
| `learning_curve` | evolucao de R2, MAE, RMSE ou outra metrica em treino/validacao |
| `validation_curve` | impacto de hiperparametro, como grau polinomial ou regularizacao |
| `coefficient_importance` | magnitude e sinal dos coeficientes de modelos lineares |
| `prediction_projection` | observado e predito ao longo de projecao PCA explicitamente solicitada |

QQ, alavancagem, Cook e erros-padrao robustos devem ser P1/P2: exigem
metadados estatisticos que um pipeline generico do scikit-learn nem sempre
fornece. Quando nao houver suporte, o relatorio deve dizer `not_available`, e
nao substituir o diagnostico por uma figura aproximada.

#### Views nao supervisionadas

Os notebooks de segmentacao e o novo
`notebooks/more/Unsupervised/dbscan.ipynb` definem o seguinte conjunto:

| View | Diagnostico e dependencias |
| --- | --- |
| `elbow_curve` | inertia/distortion por `k` para KMeans; exige candidatos KMeans |
| `silhouette_curve` | silhouette por `k`/candidato e, opcionalmente, perfil por amostra |
| `k_distance` | distancia ao k-esimo vizinho para escolher `eps` do DBSCAN; k e ponto de corte sao parametros explicitos |
| `cluster_scatter` | dispersao 2D colorida por cluster, com centroides quando disponiveis |
| `cluster_size` | barras ou pizza de tamanho dos grupos, incluindo ruido `-1` separado |
| `cluster_profile_heatmap` | medias/medianas por cluster para as features declaradas |
| `noise_outliers` | pontos marcados como ruido/anomalia por DBSCAN ou LOF |
| `dendrogram` | hierarquia de fusoes e linha de corte; exige linkage/agglomeracao com matriz de distancias |
| `pca_cluster_projection` | projecao 2D/3D dos clusters e variancia explicada |
| `pca_explained_variance` | contribuicao acumulada dos componentes usados na projecao |
| `cluster_contingency` | heatmap de cluster contra coluna de referencia fornecida, sem trata-la como target |
| `cluster_model_comparison` | silhouette, Davies-Bouldin, inertia e tamanho/ruido por candidato |

`k_distance` e um diagnostico de escolha de `eps`, nao uma curva de desempenho
do KNN. O renderer deve ordenar as distancias, informar o valor de `k` usado e
marcar o `eps` apenas quando ele tiver sido configurado pelo usuario ou
derivado por regra explicita no YAML.

#### Views de contexto do dataset

As views de contexto observadas em Titanic, California Houses, Wine, Heart
Disease e Credit Card podem existir como grupo opt-in `profile`, separado das
views que avaliam o modelo:

- `feature_distributions` — histogramas e boxplots;
- `correlation_heatmap` — correlacoes das colunas declaradas;
- `pairplot` — relacoes entre um conjunto pequeno de features;
- `target_relationships` — relacao entre features e target selecionados;
- `missingness_summary` — mapa/tabela de valores ausentes, sem preenche-los.

Essas views ajudam a contextualizar a execucao, mas nao devem selecionar o
vencedor nem ser confundidas com validacao do modelo.

### 10.1 Arquitetura do registry de views

O registry de views deve associar cada nome a:

- tarefas suportadas;
- artefatos de entrada exigidos;
- capacidades exigidas do estimador, como `predict_proba`,
  `decision_function`, `feature_importances_`, `cluster_centers_` ou
  informacoes de arvore;
- parametros aceitos e defaults seguros;
- custo estimado, principalmente quando a view executa CV adicional;
- nome e extensao dos arquivos gerados.

O runner deve terminar sua parte entregando resultados normalizados ao
`view_runner`. O `view_runner` resolve `enabled`, `scope`, capacidades e
parametros, gera as figuras e devolve status por view: `generated`, `skipped` ou
`failed`. O writer centraliza os arquivos em `reports/<experiment>/figures/`,
sem permitir que cada renderer escolha uma pasta arbitraria.

### 10.2 Fases e aceite das views

Adicionar a implementacao das views ao plano nas seguintes prioridades:

- **P0:** `model_comparison`, `confusion_matrix`, `roc_curve`,
  `predicted_vs_actual`, `residuals_vs_fitted`, `elbow_curve`,
  `silhouette_curve`, `k_distance`, `cluster_scatter`, `cluster_size` e
  `views.yaml` validado;
- **P1:** `decision_boundary`, `learning_curve`, `validation_curve`,
  `tree_structure`, `feature_importance`, `pca_cluster_projection`,
  `cluster_profile_heatmap`, `dendrogram` e `noise_outliers`;
- **P2:** `precision_recall_curve`, `probability_curve`, `qq_residuals`,
  `scale_location`, `residuals_vs_leverage`, `prediction_projection`,
  `pairplot` e testes estatisticos graficos.

Testes devem verificar que:

- cada view habilitada gera o arquivo esperado e entra no manifesto;
- uma view desabilitada nao executa treino nem gera figura;
- uma view sem capacidade ou dados suficientes vira `skipped` com motivo;
- `decision_boundary` usa a projecao declarada;
- `roc_curve` usa scores/probabilidades do split correto;
- `k_distance` registra `k`, ordenacao e `eps` quando fornecido;
- views de regressao usam somente predicoes e residuos do split correto;
- o rerun nao sobrescreve figura de outra execucao;
- views de contexto nao alteram selecao de modelo ou metricas.
