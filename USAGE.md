# Guia de uso do ML-Playground

O ML-Playground executa experimentos tabulares reproduzíveis a partir de uma tabela já preparada.
Ele valida o contrato, estima o custo, treina candidatos e persiste o pipeline vencedor.
Feature engineering, limpeza específica e parsing de domínio pertencem à receita que produz a tabela.

## Fluxo recomendado

Execute tudo a partir da raiz do repositório:

```bash
uv sync
uv run ml-playground validate --experiment iris_baseline
uv run ml-playground dry-run --experiment iris_baseline
uv run ml-playground run --experiment iris_baseline
```

`validate` lê a fonte e verifica schema, roles, fingerprint e qualidade, sem treinar modelos.
`dry-run` expande candidatos e mostra fits previstos, sem treinar modelos.
Somente `run` cria um `run_id`, persiste o campeão e publica relatórios.

Também é aceito o caminho da pasta:

```bash
uv run ml-playground validate --experiment experiments/salary_regression
uv run ml-playground run --experiment experiments/customer_segments
uv run ml-playground run --all
```

A forma legada `ml-playground --experiment <nome>` ou `ml-playground --all` continua como alias de `run`.
Não versione arquivos gerados em `reports/`, `models/` ou `logs/`.
Os comandos reais escrevem warnings, erros, resumo e duração em `logs/experiments.log`.

## Demos incluídas

As demos usam schema estrito, CV de desenvolvimento com três folds e no máximo cinco candidatos.
Elas são smoke demos: verificam o fluxo e são uma base para cópia, não um benchmark de produção.

| Experimento | Tarefa | Demonstração |
| --- | --- | --- |
| `iris_baseline` | classificação | escala numérica, baseline, KNN, logística e árvore |
| `iris_no_scaling` | classificação | contraste controlado com o Iris sem escala |
| `salary_regression` | regressão | baseline, modelos lineares e floresta pequena |
| `customer_segments` | clusterização | KMeans, DBSCAN, aglomerativo e métricas internas |

```bash
uv run ml-playground validate --experiment iris_no_scaling
uv run ml-playground dry-run --experiment salary_regression
uv run ml-playground run --experiment customer_segments
```

## Criar um experimento

```bash
uv run ml-playground init --task classification --name churn_demo
uv run ml-playground init --task regression --name price_demo --experiments-root meus_experimentos
uv run ml-playground validate --experiment churn_demo
```

As tarefas aceitas são `classification`, `regression` e `clustering`.
O template cria `experiment.yaml`, `models.yaml`, `preprocessing.yaml`, `metrics.yaml`, `cross_validation.yaml` quando aplicável e `views.yaml`.
Ele não sobrescreve arquivos existentes sem uma opção explícita.

```text
experiments/
  meu_experimento/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml   # supervisionadas
    views.yaml
```

`name` deve ser igual ao nome da pasta, e paths relativos de dados são resolvidos na raiz do projeto.

## Contrato de dados

Declare a origem, as roles e o schema antes de treinar.
Não invente uma role: se a fonte não possui chave estável, omita `id_column` e aceite que o fallback depende da ordem da fonte.

```yaml
contract_version: 1
name: churn_demo
task: classification

data:
  path: data/processed/churn.parquet
  read_options: {}
  target: churned
  id_column: customer_id             # opcional, preservado nos artefatos
  metadata_columns: [region]
  group_column: household_id         # opcional; nunca é feature
  time_column: event_at              # opcional; nunca é feature
  features:
    numeric: [age, monthly_fee, tenure]
    categorical: [plan]
  schema:
    mode: strict
    dtypes:
      customer_id: Int64
      region: String
      age: Float64
      monthly_fee: Float64
      tenure: Int64
      plan: String
      churned: Int64
  random_state: 42
```

`read_options` contém apenas opções do leitor, como separador/encoding de CSV ou sheet de Excel.
Em `schema.mode: strict`, colunas ausentes, extras ou dtypes incompatíveis impedem o primeiro fit.

- `target` é obrigatório em classificação/regressão e proibido em clusterização.
- `id_column` preserva a chave de negócio em splits, predições e inferência.
- `metadata_columns` acompanha evidências sem entrar como feature.
- `group_column` e `time_column` protegem protocolos de grupo e tempo.

`features` aceita lista legada ou grupos explícitos; prefira grupos para manter o pré-processamento tipado.

## Desenvolvimento versus teste final

Escolha modelos e hiperparâmetros exclusivamente pela evidência de desenvolvimento.
As demos supervisionadas usam CV pequena:

```yaml
evaluation:
  protocol: development
  evaluate_final_test: false
  splitter:
    name: stratified_kfold
    n_splits: 3
    shuffle: true
    random_state: 42

selection:
  primary_metric: f1_macro
  direction: maximize
  tie_breakers: [metric_std, candidate_id]
  baseline_candidate: dummy_classifier
```

Mantenha o splitter equivalente em `cross_validation.yaml` durante a janela de compatibilidade:

```yaml
method: stratified_kfold
n_splits: 3
shuffle: true
random_state: 42
```

Para regressão, use `kfold`, `group_kfold`, `group_holdout` ou `time_series` de acordo com a fonte.
Não estratifique um target contínuo sem uma regra declarada.
Para clusterização, não configure avaliação supervisionada ou teste final por padrão.

Quando houver uma fonte externa bloqueada, declare `data.test`; ela é validada antes do treino e nunca participa da seleção:

```yaml
data:
  path: data/processed/churn_development.parquet
  test:
    path: data/processed/churn_final_test.parquet
    read_options: {}
  # target, roles, features e schema continuam declarados aqui

evaluation:
  protocol: development
  evaluate_final_test: true
  splitter:
    name: stratified_kfold
    n_splits: 5
```

O manifesto separa métricas de desenvolvimento, seleção, teste final e refit de deploy.
Uma métrica do teste final nunca substitui a métrica que escolheu o candidato.

## Busca, execução e proveniência

```yaml
provenance:
  recipe_ref: recipes/prepare_churn.py
  recipe_revision: 4f3c2ab
  source_description: Snapshot tabular preparado da fonte licenciada.

search:
  strategy: grid
  max_candidates: 24

execution:
  n_jobs: 1
  max_wall_time_seconds: 900
  on_candidate_error: continue
```

`recipe_ref` é auditável, mas nunca é executada pelo core.
Use `n_jobs: 1` em exemplos e faça `dry-run` após mudar fonte, grid ou splitter.
Inclua `dummy_classifier` ou `dummy_regressor` em tarefas supervisionadas; se o baseline vencer, revise dados e protocolo antes de ampliar o grid.

## Pré-processamento e métricas

Transformações são ajustadas dentro do fold de treino:

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

Não faça encoding, imputação, normalização, seleção de feature ou PCA antes do split.
Escolha métrica escalar de desenvolvimento em `metrics.yaml` e declare direção em `selection`.
Em multiclasse, `f1_macro`, `precision_macro` e `recall_macro` complementam accuracy; ROC/log loss exigem scores ou probabilidades.
Em regressão, compare ao menos `mae`, `rmse` e `r2`; em clusterização, métricas internas não substituem validação de negócio.

## Saídas do run

```text
reports/<experiment_name>/
  metrics/<run_id>_summary.csv
  metrics/<run_id>_fold_metrics.csv
  predictions/<run_id>_predictions.csv
  predictions/<run_id>_predictions.parquet   # quando configurado
  tables/<run_id>_manifest.json
  tables/<run_id>_model_card.md
  figures/<run_id>_*.png
models/<experiment_name>/<run_id>/model.joblib
```

O manifesto contém caminhos relativos, hashes, contexto do run, fingerprint e candidato selecionado.
CSVs de resumo, folds e predições incluem `candidate_id`, `trial_id` e `split_id`; use esses IDs para relacionar evidências.
Um `row` sem `id_column` estável não é uma chave de negócio.

## Inferência segura em lote

```bash
uv run ml-playground predict \
  --model models/iris_baseline/<run_id>/model.joblib \
  --input data/processed/iris_scoring.csv \
  --output reports/iris_baseline/iris_scoring.parquet
```

`predict` carrega o artefato, valida features, tipos e ID, e preserva o ID na saída.
A saída inclui `prediction` e, para classificadores compatíveis, probabilidades por classe.

```python
import polars as pl
from ml_playground.models.inference import predict_batch

result = predict_batch(
    "models/iris_baseline/<run_id>/model.joblib",
    pl.read_csv("data/processed/iris_scoring.csv"),
    strict=True,
    output_path="reports/iris_baseline/iris_scoring.parquet",
)
print(result.frame.head())
```

KMeans pode atribuir clusters a novas linhas. DBSCAN e aglomerativo servem para reproduzir e inspecionar o ajuste, não para classificar novos clientes.

## Diagnóstico e desenvolvimento

| Situação | Ação |
| --- | --- |
| schema ou dtype falha em validate | corrija receita ou contrato; não faça cast silencioso |
| dry-run mostra muitos fits | reduza o grid ou aumente orçamento conscientemente |
| baseline vence | revise sinal, features, target e protocolo |
| ID duplicado | corrija a chave ou omita `id_column` se ela não for estável |
| inferência rejeita colunas | alinhe o lote à signature do artefato |

Para a suíte local:

```bash
uv run pytest -q --basetemp .pytest-tmp-local
```

Testes não geram logs persistentes. Veja também as [capacidades operacionais](docs/capabilities.md), [docs/upgrades.md](docs/upgrades.md), [docs/Beta.md](docs/Beta.md) e [AGENTS.md](AGENTS.md).
