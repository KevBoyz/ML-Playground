# Guia de uso

Este guia mostra como criar e executar experimentos de classificação com o
ML-Playground.

## 1. Pré-requisitos

Execute os comandos a partir da raiz do projeto. O projeto usa Python `>=3.13`
e `uv` para dependências e execução.

```bash
uv sync
```

## 2. Estrutura de um experimento

Cada experimento é uma pasta independente dentro de `experiments/`. Ela deve
conter estes cinco arquivos:

```text
experiments/
  meu_experimento/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
```

O nome da pasta deve ser igual ao campo `name` de `experiment.yaml`. O loader
não combina arquivos entre experimentos.

## 3. Configurando o experimento

### `experiment.yaml`

Define o dataset, a tarefa, a seleção do melhor modelo e as saídas:

```yaml
name: iris_baseline
task: classification

data:
  path: data/raw/iris.csv
  target: target
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

`data.path` pode ser relativo à raiz do projeto. O loader atual aceita CSV,
Parquet e planilhas. `target` precisa existir no arquivo de dados.

As direções válidas para a métrica principal são `maximize` e `minimize`.

### `models.yaml`

Lista os modelos e o grid de hiperparâmetros. Os nomes do MVP são:

- `knn`;
- `logistic_regression`;
- `svm`;
- `decision_tree`.

Cada valor em uma lista participa do produto cartesiano do grid:

```yaml
models:
  - name: knn
    params:
      n_neighbors: [3, 5, 7]
      weights: [uniform, distance]

  - name: logistic_regression
    params:
      C: [0.1, 1.0, 10.0]
      max_iter: [1000]
      random_state: [42]
```

Esse exemplo produz seis combinações de KNN e três de regressão logística.
Valores escalares também são aceitos, mas listas tornam explícito que o valor
faz parte do grid.

### `preprocessing.yaml`

O pré-processamento é separado automaticamente entre colunas numéricas e
categóricas e ajustado novamente dentro de cada fold:

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

Opções principais disponíveis no registry:

- imputação: `mean`, `median`, `mode`, `constant`, `knn`;
- escala: `standard`, `robust`, `minmax`, `none`;
- encoding: `onehot`, `ordinal`;
- transformação: `none`, `log`, `boxcox`, `yeojohnson`;
- seleção: `none`, `variance_threshold`, `mutual_info`, `f_classif`;
- dimensionalidade: `none`, `pca`, `kernel_pca`;
- outliers: `none`, `iqr`, `zscore`.

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

### `metrics.yaml`

As métricas habilitadas devem ficar na seção correspondente à tarefa:

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
  primary: f1_macro
```

Para datasets multiclasses, prefira métricas `*_macro` ou `*_weighted`. A
métrica definida em `selection.primary_metric` também precisa estar em
`names`.

### `cross_validation.yaml`

Validação cruzada estratificada:

```yaml
method: stratified_kfold
n_splits: 5
shuffle: true
random_state: 42
```

Métodos disponíveis:

- `holdout`, com `test_size` entre `0` e `1`;
- `kfold`;
- `stratified_kfold`;
- `repeated_kfold`, com `n_repeats`;
- `repeated_stratified_kfold`, com `n_repeats`.

Exemplo de holdout:

```yaml
method: holdout
test_size: 0.2
random_state: 42
```

## 4. Executando experimentos

Execute uma pasta específica:

```bash
uv run ml-playground --experiment experiments/iris_baseline
```

Também é possível informar somente o nome:

```bash
uv run ml-playground --experiment iris_baseline
```

Execute todas as subpastas válidas:

```bash
uv run ml-playground --all
```

O comando retorna código de erro quando não consegue descobrir os experimentos
ou quando nenhum modelo de um experimento executa com sucesso.

## 5. Entendendo as saídas

Cada execução recebe um `run_id`. Os artefatos são separados pelo nome do
experimento:

```text
reports/
  iris_baseline/
    metrics/
      <run_id>_summary.csv
      <run_id>_fold_metrics.csv
    tables/
      <run_id>_model_comparison.csv
      <run_id>_errors.csv
      <run_id>_predictions.csv
      <run_id>_manifest.json
    figures/
      <run_id>_metric_comparison.png
      <run_id>_roc_curves.png
      <run_id>_<modelo>_confusion_matrix.png
```

- `summary.csv`: uma linha por combinação de modelo e hiperparâmetros;
- `fold_metrics.csv`: resultado de cada fold;
- `model_comparison.csv`: valores da métrica principal;
- `errors.csv`: combinações que falharam sem interromper o grid;
- `manifest.json`: configuração efetiva, melhor resultado e caminhos dos
  artefatos;
- `predictions.csv`: valores reais, preditos e scores;
- `figures/`: gráficos gerados quando `outputs.figures` está habilitado.

O pipeline final do melhor resultado fica em:

```text
models/<experiment_name>/<run_id>/model.joblib
```

Ele inclui o pré-processamento e o estimador, portanto pode receber as colunas
brutas do dataset.

## 6. Usando o modelo persistido

```python
import polars as pl

from ml_playground.models.serialization import load_model

pipeline, metadata = load_model(
    "models/iris_baseline/<run_id>/model.joblib"
)
df = pl.read_csv("data/raw/iris.csv")
X = df.drop("target").to_pandas()
predictions = pipeline.predict(X)

print(metadata["model"])
print(predictions[:5])
```

## 7. Executando pela API Python

Para controlar o fluxo diretamente:

```python
from ml_playground.experiments.config import load_experiment
from ml_playground.experiments.runner import run_grid

config = load_experiment("experiments/iris_baseline")
result = run_grid(config, write_reports=True)

print(result["best"]["name"])
print(result["best"]["metrics"])
```

`run_grid()` usa os modelos definidos em `models.yaml`, expande o grid, captura
erros por combinação, seleciona o melhor resultado e gera os relatórios quando
`write_reports=True`.

## 8. Criando uma variante

Copie uma pasta existente, altere o nome em `experiment.yaml` e modifique os
YAMLs desejados:

```text
experiments/
  iris_baseline/
  iris_no_scaling/
  iris_com_grid_maior/
```

Por exemplo, para comparar o efeito da escala, mantenha os mesmos modelos e
altere apenas `preprocessing.yaml` para usar `method: none` em `scaling`. Ao
executar `--all`, a variante terá relatórios independentes em
`reports/iris_com_grid_maior/`.

## 9. Logs e testes

O comando real grava somente warnings/erros, resumo e tempo final em
`logs/experiments.log`. Cada execução sobrescreve o log anterior.

Para rodar os testes sem usar o diretório temporário global do Windows:

```bash
uv run pytest -q --basetemp .pytest-tmp-local
```

Os testes usam diretórios temporários e não deixam logs persistentes.
