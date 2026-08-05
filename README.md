# ML-Playground

Plataforma modular para comparar modelos de Machine Learning em experimentos
reproduzíveis.

## Execução rápida

Cada experimento é uma pasta autocontida dentro de `experiments/`:

```text
experiments/
  iris_baseline/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
    views.yaml
```

Execute um experimento com:

```bash
uv run ml-playground --experiment experiments/iris_baseline
```

Para executar todas as pastas válidas:

```bash
uv run ml-playground --all
```

O núcleo Beta suporta classificação, regressão e clusterização. Os exemplos
`iris_baseline`, `salary_regression` e `customer_segments` demonstram os três
fluxos. Cada YAML define uma parte do experimento; não há dependência de
configurações soltas ou compartilhadas.

`views.yaml` seleciona as figuras de diagnóstico por tarefa. As views P0
incluem comparação de modelos, matriz de confusão, ROC, observado versus
predito, resíduos, elbow, silhouette, k-distância, dispersão e tamanho de
clusters.

## Saídas

Cada experimento recebe uma árvore própria em `reports/`:

```text
reports/
  iris_baseline/
    metrics/    # métricas por fold e resumo
    tables/     # comparação, erros, diagnósticos tabulares e manifesto
    predictions/ # predições supervisionadas ou labels de cluster
    figures/    # views habilitadas em views.yaml
models/
  iris_baseline/<run_id>/model.joblib
logs/
  experiments.log
```

Os arquivos de saída usam o mesmo `run_id`, permitindo associar métricas,
figuras, manifesto e modelo persistido. Uma nova execução não sobrescreve uma
execução anterior.

## Estrutura do projeto

```text
data/          - Dados brutos, intermediários, processados e externos
experiments/   - Experimentos versionados, cada um com seus YAMLs
src/           - Código-fonte organizado por domínio
tests/         - Testes unitários e de integração
models/        - Pipelines treinados persistidos
reports/       - Saídas separadas por experimento
logs/          - Logs do comando real de execução
notebooks/     - Explorações interativas
docs/          - Planos e documentação de implementação
```

## Desenvolvimento

```bash
uv run pytest -q --basetemp .pytest-tmp-local
```

O roadmap está em [docs/Beta.md](docs/Beta.md). As regras de arquitetura e
logging estão em `AGENTS.md`. Consulte também o [guia de uso](USAGE.md) para
conduzir experimentos.
