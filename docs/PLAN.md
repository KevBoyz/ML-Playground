# Plano de Implementação

> Ordem sugerida para implementar cada módulo, do mais fundamental ao mais integrado.
>
> **Nota:** Todas as implementações devem seguir os nomes e estruturas definidos em `configs/`.
> Código é implementação; `configs/` é o contrato. Registry e pipeline builder
> devem refletir exatamente os métodos, categorias e parâmetros dos YAMLs.

---

## Fase 1: Fundação ✅

- [x] `ml_playground/utils/__init__.py` — `setup_logger()`, `log_summary()`, `Timer`
- [x] `ml_playground/data/loader.py` — `auto_read()` (detecção de formato)
- [x] `ml_playground/data/validation.py` — `check_missing()`, `check_dtypes()`, `validate_data()`
- [x] `ml_playground/data/cleaning.py` — `fill_missing()` (orquestração de estratégias)
- [x] **Removido vs plano original:** `split.py` (só delegava `train_test_split`), `set_seed()` (só delegava `np.random.seed`), `check_duplicates()` (só delegava `df.duplicated`)
- [x] 8 testes em `tests/`

## Fase 2: Pré-processamento ✅

- [x] `preprocessing/registry.py` — `IMPUTATION`, `SCALING`, `ENCODING`, `TRANSFORMATION`, `FEATURE_SELECTION`, `DIMENSIONALITY`, `OUTLIERS` + `get_transformer()`
- [x] `preprocessing/pipelines.py` — `build_pipeline()`, `build_column_transformer()`
- [x] Custom transformers: `IQRRemover`, `ZScoreRemover` (sklearn gap real)
- [x] **Stubs removidos:** `imputation.py`, `scaling.py`, `encoding.py`, `transformation.py`, `feature_selection.py`, `dimensionality.py`, `outliers.py` — registry único
- [x] Tests: `test_registry.py` (9), `test_pipelines.py` (6)

## Fase 3: Modelos ✅

- [x] `models/__init__.py` — `MODELS` dict (lazy loading) + `get_model()`
- [x] **Stubs removidos:** `random_forest.py`, `svm.py`, `knn.py`, `xgboost.py`, `lightgbm.py`, `logistic.py`
- [x] xgboost + lightgbm como dependências fixas (request: "uv add, nao opcionais")
- [x] Tests: `test_models.py` (10)

## Fase 4: Avaliação ✅

- [x] `evaluation/metrics.py` — `CLASSIFICATION` (11 métricas), `REGRESSION` (6 métricas), `compute_metrics()`
- [x] `evaluation/statistics.py` — `friedman_test()`, `wilcoxon_test()`, `nemenyi_test()`
- [x] **Stubs removidos:** `classification.py`, `regression.py`, `confusion.py`, `roc.py`
- [x] scipy adicionado
- [x] Tests: `test_metrics.py` (11), `test_statistics.py` (6)

## Fase 5: Experimentos ✅

- [x] `experiments/grid.py` — `expand_params()`, `build_model_grid()`
- [x] `experiments/tracker.py` — `create_run()` (ID único, dir, salva config JSON)
- [x] `experiments/executor.py` — `run_experiment()` (holdout + CV, pipeline completo)
- [x] `experiments/runner.py` — `run_grid()` (itera grid de combos, trata erros)
- [x] `experiments/comparison.py` — `compare_results()` (Friedman + Nemenyi)
- [x] Tests: `test_grid.py` (6), `test_tracker.py` (3), `test_executor.py` (3), `test_runner.py` (2), `test_comparison.py` (2)

## Fase 6: Visualização ✅

- [x] `visualization/barplots.py` — `plot_metric_comparison()`
- [x] `visualization/heatmaps.py` — `plot_confusion_matrix()`, `plot_correlation_heatmap()`
- [x] `visualization/roc.py` — `plot_roc_curves()` (multi-model)
- [x] `visualization/importance.py` — `plot_feature_importance()`
- [x] `visualization/comparison.py` — `plot_cd_diagram()`
- [x] **Stub removido:** `confusion.py` (absorvido em heatmaps)
- [x] matplotlib adicionado
- [x] Tests: `test_visualization.py` (11)

## Fase 7: Relatórios e Automação ⏭️

- [ ] Skip — user pulou para Fase 9
- [ ] Relatório CSV implementado em Fase 9 (`experiments/report.py`)

## Fase 8: Testes e Qualidade ⏭️

- [ ] Skip — user pulou para Fase 9
- [ ] 94 testes passando cobrem todas as fases implementadas

## Fase 9: Expansões ✅

- [x] `models/serialization.py` — `save_model()` / `load_model()` com metadados (joblib)
- [x] `experiments/cache.py` — `PipelineCache` (cache de preprocessing por hash MD5)
- [x] `experiments/report.py` — `results_to_csv()` (relatório CSV de experimentos)
- [ ] Otimização Bayesiana / AutoML / MLflow / ONNX — próximo ciclo

---

**Total: 94 testes, 0 falhas.**
