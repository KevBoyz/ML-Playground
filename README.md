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
```

Execute um experimento com:

```bash
uv run ml-playground --experiment experiments/iris_baseline
```

Para executar todas as pastas válidas:

```bash
uv run ml-playground --all
```

O MVP suporta KNN, regressão logística, SVM e árvore de decisão. Cada YAML
define uma parte do experimento; não há dependência de configurações soltas ou
compartilhadas.

## Saídas

Cada experimento recebe uma árvore própria em `reports/`:

```text
reports/
  iris_baseline/
    metrics/    # métricas por fold e resumo
    tables/     # comparação, erros, predições e manifesto
    figures/    # comparação e matriz de confusão
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

O fluxo completo está detalhado em [docs/PLAN.md](docs/PLAN.md). As regras de
arquitetura e logging estão em `AGENTS.md`. Consulte também o
[guia de uso](docs/USAGE.md) para conduzir experimentos.
