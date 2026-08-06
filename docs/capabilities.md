# Capacidades operacionais

Este documento descreve o que o repositório executa hoje. O plano completo e
suas decisões de escopo ficam em [upgrades.md](upgrades.md).

| Área | Disponível | Limite deliberado |
| --- | --- | --- |
| Tarefas | Classificação, regressão e clusterização tabular | Não há deep learning, download de dados ou AutoML. |
| Dados | CSV, Parquet e planilhas com `read_options`; schema, ID, metadados, grupo, tempo e fonte de teste externa | A receita que produz a tabela continua fora do core. |
| Preflight | Profile, fingerprint, assinatura de schema, validação de roles e contrato antes do primeiro fit | Não corrige nem faz casts silenciosos. |
| Avaliação supervisionada | Holdout, CV, repeated CV, grupo, tempo/backtest, teste externo/split reservado e nested CV | Clusterização usa métricas internas; não recebe teste final por padrão. |
| Seleção | Métrica de desenvolvimento, direção, desempate determinístico e baseline opcional | O teste final é avaliado uma única vez depois da seleção. |
| Busca | Grid ou amostra aleatória limitada por `max_candidates`, orçamento de tempo e política de erro | Execução por candidatos é sequencial para evitar paralelismo aninhado. |
| Evidências | IDs de candidato/trial/split, plano de splits, previsões OOF/finais, manifest relativo com checksums e model card | `resume` e comparação histórica entre runs ainda não são comandos públicos. |
| Artefatos | Pipeline completo, signature de entrada, proveniência e contexto do ambiente | DBSCAN e aglomerativo não pontuam novas linhas. |
| Inferência | `predict`/`predict_batch` com validação de features, tipos e ID; saída CSV ou Parquet | Não há serving HTTP. |
| CLI | `validate`, `dry-run`, `run`, `init` e `predict` | A API Python continua adequada para integração programática. |

## Fluxo de decisão

```text
tabela preparada
  -> validate / preflight
  -> dry-run (candidatos e fits)
  -> desenvolvimento (seleção)
  -> teste final bloqueado, se configurado
  -> refit de deploy, artefato e relatório
  -> predict em lote com signature validada
```

Use [USAGE.md](../USAGE.md) para exemplos completos de YAML e comandos.
