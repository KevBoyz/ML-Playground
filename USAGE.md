# Guia completo de uso do ML-Playground

O ML-Playground executa experimentos tabulares reproduzíveis por configuração.
Ele lê um dataset já preparado, compara modelos e hiperparâmetros, seleciona o
melhor candidato, salva o pipeline treinado e publica métricas, predições,
tabelas e views de diagnóstico.

As tarefas atualmente operacionais são:

- classificação binária e multiclasse;
- regressão com target numérico;
- clusterização sem target supervisionado.

O pacote coordena treinamento e avaliação. Decisões de domínio — download de
dados, parsing de strings, feature engineering, agregações de negócio e limpeza
específica — devem ocorrer antes de o dataset ser entregue ao experimento.

## 1. Pré-requisitos e comandos básicos

Execute os comandos a partir da raiz do projeto:

```bash
uv sync
uv run ml-playground --experiment iris_baseline
```

Também é possível informar o caminho completo da pasta:

```bash
uv run ml-playground --experiment experiments/salary_regression
```

Para executar todos os experimentos descobertos em `experiments/`:

```bash
uv run ml-playground --all
```

Para usar outra raiz de experimentos:

```bash
uv run ml-playground --all --experiments-root meus_experimentos
```

O comando retorna `0` somente quando processa ao menos um experimento e nenhum
deles falha. Warnings, erros, resumo e tempo total da execução são gravados em
`logs/experiments.log`, substituindo o log da execução anterior.

## 2. Exemplos incluídos

| Experimento | Tarefa | Dataset | Demonstra |
| --- | --- | --- | --- |
| `iris_baseline` | classificação multiclasse | `data/raw/iris.csv` | KNN, regressão logística, SVM, árvore, CV estratificada, matriz de confusão e ROC |
| `iris_no_scaling` | classificação multiclasse | `data/raw/iris.csv` | comparação da variante sem escala numérica |
| `salary_regression` | regressão | `data/raw/salary_regression.csv` | baseline, regressões linear/regularizada, floresta, resíduos e observado versus predito |
| `customer_segments` | clusterização | `data/raw/customer_segments.csv` | KMeans, DBSCAN, aglomerativo, silhouette, elbow e dispersão de clusters |

Execute qualquer um deles pelo nome:

```bash
uv run ml-playground --experiment customer_segments
```

## 3. Estrutura de um experimento

Um experimento é uma pasta autocontida. Os quatro primeiros arquivos são
obrigatórios; `cross_validation.yaml` e `views.yaml` são opcionais conforme a
tarefa.

```text
experiments/
  meu_experimento/
    experiment.yaml          # obrigatório: tarefa, dataset, saídas e seleção
    models.yaml              # obrigatório: candidatos e grids
    preprocessing.yaml       # obrigatório: pipeline genérico ou {}
    metrics.yaml             # obrigatório: métricas da tarefa
    cross_validation.yaml    # opcional para supervisionadas; não usar em clustering
    views.yaml               # opcional: gráficos diagnósticos
```

Regras de nome e isolamento:

- o nome da pasta e `experiment.yaml.name` devem ser idênticos;
- o nome aceita letras, números, `_` e `-`;
- não há herança entre pastas de experimento;
- caminhos relativos em `data.path` e `outputs.root` são resolvidos a partir
  da raiz do projeto;
- um erro em um candidato do grid é registrado, mas não impede os demais;
- uma execução falha quando nenhum candidato produz a métrica de seleção.

## 4. Contrato dos dados

### Formatos aceitos

O loader aceita:

- CSV: `.csv`;
- Parquet: `.parquet`;
- planilhas: `.xlsx`, `.xls` e `.ods`.

O dataset é lido por Polars e convertido para Pandas antes de entrar no
pipeline do scikit-learn.

### Colunas e features

Para classificação e regressão, `data.target` é obrigatório. `data.features`
é opcional; se omitido, todas as colunas exceto o target serão usadas.

Para clusterização, `data.target` é proibido e `data.features` é obrigatório.
Isso evita que uma coluna de identificação ou referência entre no modelo sem
uma escolha explícita.

O pacote valida que as colunas declaradas existem antes de chamar `fit`. Ele
não remove identificadores, não cria features e não corrige tipos sozinho.

## 5. `experiment.yaml`

Este arquivo define a tarefa, o dataset, as saídas e como escolher o vencedor.

### Classificação

```yaml
name: churn_classification
task: classification

data:
  path: data/raw/churn.csv
  target: churned
  features: [age, monthly_fee, tenure, plan]
  random_state: 42

outputs:
  root: reports
  save_model: true
  save_predictions: true
  figures: true

selection:
  primary_metric: f1_macro
  direction: maximize
```

### Regressão

```yaml
name: price_regression
task: regression

data:
  path: data/raw/prices.parquet
  target: price
  random_state: 42

outputs:
  root: reports
  save_model: true
  save_predictions: true
  figures: true

selection:
  primary_metric: rmse
  direction: minimize
```

### Clusterização

```yaml
name: customer_segments
task: clustering

data:
  path: data/raw/customers.csv
  features: [annual_income, spending_score]
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

### Campos de saída

| Campo | Padrão | Efeito |
| --- | --- | --- |
| `outputs.root` | `reports` | raiz dos relatórios |
| `outputs.save_model` | `true` | salva o pipeline do melhor candidato em `models/` |
| `outputs.save_predictions` | `true` | salva predições supervisionadas ou labels de cluster |
| `outputs.figures` | `true` | permite renderizar as views habilitadas |

Quando `figures: false`, as views aparecem como `skipped` no manifesto e
nenhuma figura é criada.

### Regra de seleção

`selection.primary_metric` deve estar em `metrics.yaml`. As direções válidas
são `maximize` e `minimize`.

As direções padrão são `minimize` para `mae`, `mse`, `rmse`, `mape`,
`max_error` e `davies_bouldin`; para as demais métricas o padrão é
`maximize`. Ainda assim, declare a direção explicitamente para deixar a
intenção do experimento clara.

Não use `confusion_matrix` como métrica principal: ela é uma matriz, não um
valor escalar comparável.

## 6. `models.yaml`: modelos e grids

Cada item de `models` possui um nome do registry e um mapa `params`. Listas
formam o produto cartesiano do grid; valores escalares também são aceitos.
`params: {}` cria exatamente um candidato com os parâmetros padrão do
estimador.

```yaml
models:
  - name: ridge
    params:
      alpha: [0.1, 1.0, 10.0]

  - name: random_forest_regressor
    params:
      n_estimators: [100, 300]
      max_depth: [null, 10]
      random_state: [42]
```

Os nomes dos parâmetros são os mesmos dos construtores do scikit-learn (ou da
biblioteca do estimador). Um parâmetro inválido faz apenas aquele candidato
aparecer em `errors.csv`.

### Modelos por tarefa

| Tarefa | Nomes aceitos |
| --- | --- |
| Classificação | `knn`, `logistic_regression`, `logistic` (alias), `svm`, `decision_tree`, `random_forest`, `xgboost`, `lightgbm` |
| Regressão | `dummy_regressor`, `linear_regression`, `ridge`, `lasso`, `elastic_net`, `random_forest_regressor` |
| Clusterização | `kmeans`, `dbscan`, `agglomerative` |

`xgboost` e `lightgbm` são importados apenas quando solicitados. Se a
dependência não estiver instalada, a combinação falha e o motivo é publicado
no relatório.

### Exemplo de classificação

```yaml
models:
  - name: knn
    params:
      n_neighbors: [3, 5, 7]
      weights: [uniform, distance]

  - name: logistic_regression
    params:
      C: [0.1, 1.0]
      max_iter: [1000]
      random_state: [42]

  - name: svm
    params:
      kernel: [linear, rbf]
      C: [1.0]
      probability: [true]

  - name: decision_tree
    params:
      max_depth: [null, 5]
      random_state: [42]
```

### Exemplo de regressão

```yaml
models:
  - name: dummy_regressor
    params:
      strategy: [mean]

  - name: linear_regression
    params: {}

  - name: elastic_net
    params:
      alpha: [0.01, 0.1]
      l1_ratio: [0.2, 0.8]
      max_iter: [5000]
```

Use `dummy_regressor` como baseline: ele ajuda a verificar se o modelo real
supera uma previsão constante.

### Exemplo de clusterização

```yaml
models:
  - name: kmeans
    params:
      n_clusters: [2, 3, 4, 5]
      n_init: [10]
      random_state: [42]

  - name: dbscan
    params:
      eps: [0.3, 0.5, 0.7]
      min_samples: [5]

  - name: agglomerative
    params:
      n_clusters: [2, 3, 4]
      linkage: [ward]
```

DBSCAN pode retornar ruído, identificado pelo label `-1`. O pacote não trata
esses pontos como um cluster comum ao calcular as métricas internas.

## 7. `preprocessing.yaml`: pipeline explícito

`preprocessing.yaml` é obrigatório, mas pode conter somente `{}` quando os
dados já estiverem prontos para os modelos. O padrão é passthrough: o pacote
não aplica normalização, imputação ou encoding implicitamente.

O formato recomendado separa as colunas numéricas e categóricas. Cada pipeline
é ajustado dentro do fold de treino, evitando leakage em validação cruzada.

```yaml
numeric:
  steps:
    - name: imputation
      category: imputation
      method: median
    - name: scaling
      category: scaling
      method: standard

categorical:
  steps:
    - name: imputation
      category: imputation
      method: mode
    - name: encoding
      category: encoding
      method: onehot
```

### Categorias e métodos disponíveis

| Categoria | Métodos |
| --- | --- |
| `imputation` | `mean`, `median`, `mode`, `constant`, `knn` |
| `scaling` | `standard`, `robust`, `minmax`, `none` |
| `encoding` | `onehot`, `ordinal` |
| `transformation` | `none`, `log`, `boxcox`, `yeojohnson` |
| `feature_selection` | `none`, `variance_threshold`, `mutual_info`, `f_classif` |
| `dimensionality` | `none`, `pca`, `kernel_pca` |
| `outliers` | `none`, `iqr`, `zscore` |

Parâmetros específicos ficam em `params`:

```yaml
numeric:
  steps:
    - name: imputation
      category: imputation
      method: knn
      params:
        n_neighbors: 5
```

Cuidados:

- `boxcox` exige valores positivos;
- `mutual_info` e `f_classif` são seletores voltados à classificação;
- PCA e KernelPCA mudam o espaço de features usado pelo modelo;
- clipping por `iqr` e `zscore` altera valores extremos, não remove linhas;
- use apenas transformações cuja semântica você conhece para o seu dataset.

## 8. `metrics.yaml`: avaliação

As métricas devem ficar na seção da tarefa. `primary` é opcional, mas evita
duplicar a métrica de seleção em `experiment.yaml`.

### Classificação

```yaml
classification:
  names:
    - accuracy
    - precision_macro
    - recall_macro
    - f1_macro
    - f1_weighted
    - roc_auc
    - log_loss
    - kappa
    - mcc
  primary: f1_macro
```

Métricas aceitas: `accuracy`, `precision`, `precision_macro`,
`precision_weighted`, `recall`, `recall_macro`, `recall_weighted`, `f1`,
`f1_macro`, `f1_weighted`, `roc_auc`, `log_loss`, `kappa`, `mcc` e
`confusion_matrix`.

Para multiclasses, prefira `*_macro` ou `*_weighted`. `roc_auc` requer
probabilidades ou score; `log_loss` requer probabilidades. Para SVM, use
`probability: true` quando desejar probabilidades explícitas.

### Regressão

```yaml
regression:
  names: [mae, rmse, r2, mape]
  primary: rmse
```

Métricas aceitas: `mae`, `mse`, `rmse`, `r2`, `mape` e `max_error`.

### Clusterização

```yaml
clustering:
  names:
    - silhouette
    - davies_bouldin
    - inertia
    - cluster_count
    - noise_ratio
  primary: silhouette
```

Métricas aceitas:

- `silhouette` — maior é melhor;
- `calinski_harabasz` — maior é melhor;
- `davies_bouldin` — menor é melhor;
- `inertia` — disponível para estimadores que expõem `inertia_`, como KMeans;
- `cluster_count` — quantidade de clusters sem contar ruído;
- `noise_ratio` — fração de labels `-1`;
- `cluster_size_min` e `cluster_size_max` — menor e maior grupo.

Silhouette, Calinski-Harabasz e Davies-Bouldin exigem entre dois e `n - 1`
clusters não-ruído. Quando isso não ocorre, a métrica fica indisponível com o
motivo em `metric_notes`; ela não é substituída silenciosamente por zero.

Clusterização não possui acurácia sem labels externos e, por isso, não aceita
métricas supervisionadas.

## 9. `cross_validation.yaml`: splits supervisionados

Este arquivo é opcional para classificação e regressão; sem ele, o padrão é
holdout com `test_size: 0.2`. Para clusterização, omita o arquivo. Caso ele
exista, somente `method: none` é aceito.

### Classificação

```yaml
method: stratified_kfold
n_splits: 5
shuffle: true
random_state: 42
```

Métodos aceitos:

- `holdout`, com `test_size` entre `0` e `1`;
- `kfold`;
- `stratified_kfold`;
- `repeated_kfold`, com `n_repeats`;
- `repeated_stratified_kfold`, com `n_repeats`.

Em classificação, `kfold` é promovido a `stratified_kfold` por padrão. O
holdout também usa estratificação por padrão; use `stratified: false` apenas
quando houver um motivo válido para não preservar a distribuição de classes.

### Regressão

```yaml
method: repeated_kfold
n_splits: 5
n_repeats: 3
random_state: 42
```

Regressão aceita `holdout`, `kfold` e `repeated_kfold`. Métodos estratificados
são rejeitados nessa tarefa.

## 10. `views.yaml`: diagnósticos visuais

Views são figuras opcionais geradas após a seleção. Cada item possui:

- `name` — nome do renderer;
- `enabled` — `true` ou `false`;
- `scope` — `best`, `candidates`, `selected_models` ou `folds`;
- `params` — parâmetros específicos da view.

As views são agrupadas em `common` e na chave da tarefa:

```yaml
views:
  common:
    - name: model_comparison
      enabled: true
      scope: candidates

  regression:
    - name: predicted_vs_actual
      enabled: true
      scope: best
    - name: residuals_vs_fitted
      enabled: true
      scope: best
```

O manifesto registra cada view como `generated`, `skipped` ou `failed`. Uma
view incompatível não impede a criação de métricas, modelos ou outras figuras.

### Views implementadas

| View | Tarefa | Escopo operacional | O que produz |
| --- | --- | --- | --- |
| `model_comparison` | todas | `candidates` | barras da métrica principal por candidato |
| `confusion_matrix` | classificação | `best` | matriz de erro; `params.normalize: true` normaliza por classe real |
| `roc_curve` | classificação | `best` | curva ROC a partir de score/probabilidade |
| `predicted_vs_actual` | regressão | `best` | dispersão observado versus predito e diagonal perfeita |
| `residuals_vs_fitted` | regressão | `best` | resíduos contra valores ajustados |
| `elbow_curve` | clusterização | `candidates` | inertia por `n_clusters` dos candidatos KMeans |
| `silhouette_curve` | clusterização | `candidates` | silhouette por `n_clusters` dos candidatos KMeans |
| `k_distance` | clusterização | `best` | distâncias ordenadas ao k-ésimo vizinho; útil para escolher `eps` do DBSCAN |
| `cluster_scatter` | clusterização | `best` | dispersão das duas primeiras features ou de `params.features` |
| `cluster_size` | clusterização | `best` | barras de tamanho por cluster, com ruído `-1` separado |

Exemplo para DBSCAN:

```yaml
views:
  clustering:
    - name: k_distance
      enabled: true
      scope: best
      params:
        n_neighbors: 5
        eps: 0.55
```

`k_distance` usa o espaço após o pré-processamento do vencedor. Ele apenas
mostra o `eps` declarado; não escolhe automaticamente o parâmetro.

### Views aceitas, mas ainda não renderizadas

O arquivo é validado para as views abaixo, porém a implementação atual as
registra como `skipped` com motivo `not_implemented`:

- classificação: `precision_recall_curve`, `class_distribution`,
  `decision_boundary`, `probability_curve`, `learning_curve`,
  `validation_curve`, `knn_neighbors_curve`, `tree_structure` e
  `feature_importance`;
- regressão: `fit_vs_feature`, `residual_distribution`, `qq_residuals`,
  `scale_location`, `residuals_vs_leverage`, `coefficient_importance`,
  `prediction_projection`, `learning_curve` e `validation_curve`;
- clusterização: `cluster_profile_heatmap`, `noise_outliers`, `dendrogram`,
  `pca_cluster_projection`, `pca_explained_variance`, `cluster_contingency` e
  `cluster_model_comparison`;
- contexto: `feature_distributions`, `correlation_heatmap`, `pairplot`,
  `target_relationships` e `missingness_summary`.

Esses nomes constam do roadmap em `docs/Beta.md`; não os habilite esperando
uma imagem no estado atual do projeto.

## 11. Saídas e como interpretá-las

Cada execução possui um `run_id` e fica isolada em:

```text
reports/
  <experiment_name>/
    metrics/
      <run_id>_summary.csv
      <run_id>_fold_metrics.csv
    tables/
      <run_id>_model_comparison.csv
      <run_id>_errors.csv             # somente quando houver candidatos com erro
      <run_id>_manifest.json
      <run_id>_<model>_confusion_matrix.csv  # classificação
      <run_id>_<model>_residuals.csv         # regressão
      <run_id>_<model>_cluster_sizes.csv     # clusterização
    predictions/
      <run_id>_predictions.csv
    figures/
      <run_id>_*.png

models/
  <experiment_name>/<run_id>/model.joblib
```

| Artefato | Conteúdo |
| --- | --- |
| `summary.csv` | uma linha por candidato, parâmetros, duração, métricas e erros |
| `fold_metrics.csv` | métricas e tamanhos por fold; pode estar vazio em holdout ou clusterização |
| `model_comparison.csv` | valor da métrica principal para cada candidato válido |
| `errors.csv` | candidatos que falharam, sem abortar o grid inteiro; não é criado quando não há erros |
| `predictions.csv` | `y_true`, `y_pred` e scores supervisionados, ou `cluster` por linha |
| tabelas específicas | matriz de confusão, resíduos ou tamanhos dos clusters do vencedor |
| `manifest.json` | configuração efetiva, melhor resultado, caminhos dos artefatos e status das views |

Não reutilize `row` de `predictions.csv` como identificador de negócio: ele é a
posição da linha no dataset entregue ao experimento.

## 12. Carregando o modelo persistido

O artefato salvo contém o pipeline completo, incluindo pré-processamento e
estimador. Para tarefas supervisionadas:

```python
import polars as pl

from ml_playground.models.serialization import load_model

pipeline, metadata = load_model(
    "models/iris_baseline/<run_id>/model.joblib"
)

frame = pl.read_csv("data/raw/iris.csv")
features = frame.drop("target").to_pandas()
predictions = pipeline.predict(features)

print(metadata["task"])
print(metadata["model"])
print(predictions[:5])
```

Para KMeans, o pipeline persistido também suporta `predict` em novas linhas.
DBSCAN e aglomerativo não oferecem, de forma geral, atribuição de cluster para
novos pontos; o artefato serve para reprodutibilidade e inspeção do ajuste que
gerou os labels publicados.

## 13. API Python

O fluxo usual pela API é equivalente ao CLI:

```python
from ml_playground.experiments.config import load_experiment
from ml_playground.experiments.runner import run_grid

config = load_experiment("experiments/salary_regression")
result = run_grid(config, write_reports=True)

best = result["best"]
print(best["name"])
print(best["metrics"])
print(result["reports"]["manifest"])
```

`run_grid()` expande os grids, captura erros por candidato, aplica a regra de
seleção, persiste o melhor pipeline e gera relatórios quando
`write_reports=True`.

Para executar uma única configuração já normalizada:

```python
from ml_playground.experiments.executor import run_experiment

result = run_experiment(
    {
        "task": "regression",
        "data": {"path": "data/raw/salary_regression.csv", "target": "annual_salary"},
        "model": {"name": "linear_regression", "params": {}},
        "metrics": ["mae", "rmse", "r2"],
    }
)
print(result["metrics"])
```

Esse caminho não compõe YAMLs, não cria `run_id` e não escreve relatórios por
padrão; é mais adequado para testes ou integração por código.

## 14. Criando uma variante

Copie uma pasta existente e altere somente o necessário:

```text
experiments/
  salary_regression/
  salary_regression_without_scaling/
```

Depois ajuste `experiment.yaml.name` para o nome da nova pasta. Por exemplo,
para avaliar o impacto de escala em modelos lineares, troque:

```yaml
method: standard
```

por:

```yaml
method: none
```

As duas pastas terão relatórios e modelos separados. Você também pode criar
uma variante que altera somente `models.yaml`, `metrics.yaml`, validação ou
views, mantendo o mesmo dataset.

## 15. Diagnóstico e solução de problemas

| Sintoma | Causa provável | Ação |
| --- | --- | --- |
| `Dataset não encontrado` | caminho relativo resolvido a partir da raiz do projeto | corrija `data.path` ou use caminho absoluto |
| `Target não encontrada` | nome da coluna não existe no arquivo | revise `data.target` e o cabeçalho do dataset |
| `Features não encontradas` | lista em `data.features` não corresponde às colunas | corrija os nomes declarados |
| nenhum vencedor | todos os candidatos falharam ou não produziram a métrica principal | se existir, abra `errors.csv`; reduza o grid e confira compatibilidade de modelo/métrica |
| ROC ausente | estimador não expõe score/probabilidade ou não há dados válidos | habilite `probability` no SVM quando necessário e verifique `manifest.json` |
| silhouette indisponível | DBSCAN gerou apenas ruído/um grupo ou um label por linha | ajuste `eps`, `min_samples`, features ou escala |
| `cluster_scatter` ignorada | não há exatamente duas features selecionáveis | informe `params.features: [coluna_x, coluna_y]` |
| figura não criada | view está desabilitada, não implementada ou incompatível | consulte o array `views` em `manifest.json` |

## 16. Testes e desenvolvimento

Execute a suíte completa sem depender do diretório temporário global do
Windows:

```bash
uv run pytest -q --basetemp .pytest-tmp-local
```

Os testes não geram logs persistentes. A suíte cobre os três runners, seleção,
persistência, relatórios e views P0.

## 17. Limites atuais e roadmap

O projeto não baixa datasets, não executa notebooks, não faz AutoML, não
suporta deep learning ou séries temporais especializadas e não automatiza
decisões de pré-processamento de domínio.

Mais diagnósticos, modelos e análises estatísticas declarativas estão descritos
em [docs/Beta.md](docs/Beta.md). Antes de configurar uma view, consulte a
seção "Views aceitas, mas ainda não renderizadas" deste guia para diferenciar
o que já funciona do que está planejado.
