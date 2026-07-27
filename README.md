# ML-Playground

Plataforma modular de experimentação em Machine Learning.

## Estrutura

```
data/          - Dados brutos, intermediários, processados e externos
configs/       - Configurações YAML para experimentos
src/           - Código-fonte organizado por domínio
  data/        - Carregamento, limpeza, divisão e validação
  preprocessing/ - Imputação, outliers, transformação, scaling, encoding, seleção
  models/      - Instanciadores de modelos (RF, SVM, KNN, XGB, LGB, Logistic)
  evaluation/  - Métricas, classificação, regressão, estatística
  experiments/ - Runner, executor, tracker, grid search, comparação
  visualization/ - Gráficos e plots
  utils/       - Utilitários diversos
tests/         - Testes unitários
models/        - Modelos treinados salvos
reports/       - Relatórios, figuras, métricas, tabelas
notebooks/     - Jupyter notebooks
```

## Princípios

- Componentes independentes e reutilizáveis
- Configuração via YAML, não código
- Registries para desacoplamento
- Reprodutibilidade com seeds e versionamento
- Experimentos automatizados em grid

## Próximos Passos

Ver `docs/PLAN.md`.
