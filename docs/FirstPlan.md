# Plano de próximos passos — MVP de experimentação classificatória

## 1. Objetivo

Transformar os componentes já existentes em um fluxo reproduzível, acionável por
configuração, para comparar modelos de classificação dentro de pipelines de
pré-processamento. A primeira versão deve executar um experimento completo a
partir de uma pasta em `experiments/`, comparar as combinações definidas e
publicar todas as saídas em uma pasta homônima dentro de `reports/`.

Escopo do MVP:

- KNN;
- regressão logística (`LogisticRegression`);
- SVM;
- árvore de decisão (`DecisionTreeClassifier`);
- holdout e validação cruzada estratificada;
- busca em grid de hiperparâmetros;
- métricas de classificação para problemas binários e multiclasses;
- artefatos de métricas, tabelas, figuras, configuração e modelo treinado;
- logs de execução conforme `AGENTS.md`.

Random Forest, XGBoost e LightGBM permanecem como extensões posteriores. O fato
de já existirem no registry não os torna parte do critério de pronto do MVP.

## Status desta implementação

- [x] Pastas autocontidas em `experiments/`, com dois exemplos do Iris.
- [x] Loader/validador dos cinco YAMLs e CLI para um experimento ou `--all`.
- [x] KNN, regressão logística, SVM e árvore de decisão no fluxo de grid.
- [x] Pipeline único com separação numérica/categórica e pré-processamento por
  fold.
- [x] Holdout, K-Fold, Stratified K-Fold e variantes repetidas.
- [x] Métricas multiclasses, seleção do melhor modelo e pipeline persistido.
- [x] Relatórios isolados em `reports/<experiment_name>/` e manifesto por run.
- [x] README, logs de execução e 97 testes automatizados.

Pendências secundárias para ciclos seguintes: implementar `val_size` como uma
partição distinta, integrar o cache de pré-processamento ao executor, ampliar
as visualizações de importância de features e adicionar novos modelos além do
MVP.

## 2. Diagnóstico do estado atual

### O que já existe

- `data/loader.py` lê CSV, Parquet e planilhas.
- `data/validation.py` e `data/cleaning.py` fornecem validações e estratégias
  básicas de valores ausentes.
- `preprocessing/registry.py` e `preprocessing/pipelines.py` oferecem um
  registry e um builder para transformadores.
- `experiments/grid.py`, `executor.py` e `runner.py` já esboçam expansão de
  parâmetros, holdout e K-Fold.
- `evaluation/metrics.py` contém um registry de métricas, e os módulos de
  visualização já produzem figuras.
- `models/serialization.py` e `experiments/cache.py` fornecem blocos para
  persistência e cache.
- O dataset de exemplo é `data/raw/iris.csv`, com 150 linhas, quatro atributos
  numéricos e a coluna `target`.
- Antes desta implementação, a suíte tinha 94 testes passando quando o
  diretório temporário do pytest era colocado dentro do workspace. O ambiente
  bloqueia o diretório temporário global do Windows; após a implementação, a
  suíte passou a ter 97 testes, sem criar logs persistentes para os testes.

### Lacunas identificadas antes da implementação

1. Não há carregador de YAML, descoberta de experimentos nem comando de
   execução. Hoje `run_grid()` recebe dicionários já montados por código.
2. Os YAMLs ainda estão soltos em `configs/`; eles precisam ser agrupados em
   pastas autocontidas dentro de `experiments/`, cada pasta representando uma
   execução/variante de experimento.
3. O experimento de exemplo aponta para `data/raw/dataset.csv`, que não existe;
   deve apontar para `data/raw/iris.csv` e declarar explicitamente o tipo de
   tarefa e a estratégia de seleção.
4. Os arquivos de modelos, o registry e os testes usam `logistic`, enquanto o
   contrato do projeto precisa representar claramente regressão logística.
   Também falta `decision_tree`/`DecisionTreeClassifier`.
5. `cross_validation.yaml` lista `stratified_kfold` e `repeated_kfold`, mas o
   executor trata qualquer método diferente de holdout como K-Fold comum e não
   usa a lista de métodos do arquivo.
6. `val_size` e outros campos dos YAMLs são ignorados; não existe validação de
   schema, de colunas, de target, de compatibilidade entre tarefa, modelo e
   métricas, nem falha rápida para configuração inválida.
7. O pré-processamento e o estimador são ajustados separadamente. O artefato
   salvo atualmente contém apenas o modelo, não o pipeline capaz de receber os
   dados brutos. O `ColumnTransformer` existente não é usado pelo executor.
8. Há opções declaradas que não estão implementadas ou não estão alinhadas ao
   registry: por exemplo, `transformation.log` é `None`, e `boxcox`/`yeojohnson`
   não distinguem o método escolhido.
9. `precision`, `recall` e `f1` usam o padrão binário e podem falhar no Iris;
   `roc_auc`/`log_loss` só extraem a segunda coluna de probabilidades e não
   tratam multiclasses ou SVM sem `predict_proba`.
10. O runner captura erros, mas não os publica em um relatório consolidado. O
   tracker cria `runs/` na raiz e não conecta o diretório da execução a
   `reports/`; `results_to_csv()` só grava onde o chamador mandar.
11. Não existe contrato para nomes de arquivos, separação entre métricas,
    tabelas e figuras, seleção do melhor modelo, salvamento de predições ou
    geração de figuras a partir dos resultados.
12. Não há um script real usando `setup_logger()`. Além disso, o logger atual
    grava todos os `INFO`, enquanto a lei do projeto exige apenas erros/warnings,
    resumo ao final e tempo de execução na última linha.
13. `run_grid()` está anotado como retornando lista, mas retorna um dicionário;
    o contrato público precisa ser corrigido antes de ser usado pela CLI.

## 3. Contrato alvo de experimentos

`experiments/` será a raiz dos experimentos versionados. Cada subpasta é uma
unidade independente, com nome próprio, todos os YAMLs necessários e nenhuma
dependência implícita de um catálogo global:

```text
experiments/
  iris_baseline/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
  iris_without_scaling/
    experiment.yaml
    models.yaml
    preprocessing.yaml
    metrics.yaml
    cross_validation.yaml
```

Papéis dos arquivos dentro de cada experimento:

- `experiment.yaml`: nome, tarefa, dataset, target, seed, regra de seleção e
  opções de saída;
- `models.yaml`: candidatos e grids de hiperparâmetros daquela execução;
- `preprocessing.yaml`: pipeline e opções de pré-processamento daquela
  execução;
- `metrics.yaml`: métricas habilitadas e, quando aplicável, a métrica principal;
- `cross_validation.yaml`: método e parâmetros de validação daquela execução.

O loader deve receber uma pasta, por exemplo
`experiments/iris_baseline/`, descobrir os cinco arquivos obrigatórios, resolver
caminhos relativos à raiz do projeto e produzir uma configuração normalizada.
Também pode aceitar `--all` para descobrir todas as subpastas válidas de
`experiments/`, mas uma execução individual deve ser sempre possível.

Não haverá herança silenciosa entre experimentos. Defaults internos podem ser
aplicados apenas quando documentados no schema; overrides de CLI devem ser
explícitos e limitados ao experimento selecionado. O nome da pasta será o
`experiment_name`, validado contra traversal e usado para localizar todos os
artefatos da execução.

O contrato mínimo de `experiments/iris_baseline/experiment.yaml` deve conter:

```yaml
task: classification

name: iris_baseline

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

Os demais campos desse contrato ficam nos YAMLs irmãos. Por exemplo,
`models.yaml` não deve apontar para `configs/models.yaml`; ele deve conter
diretamente os modelos e parâmetros do experimento.

Os nomes canônicos do registry devem ser `knn`, `logistic_regression`, `svm` e
`decision_tree`. `logistic` pode ser mantido apenas como alias de compatibilidade
durante a migração, mas não deve aparecer nos novos exemplos.

## 4. Plano de implementação

### Fase 1 — Contrato, loader e comando de execução (P0)

- Migrar os YAMLs atuais de `configs/` para
  `experiments/iris_baseline/`, completando os campos do contrato, e remover a
  fonte duplicada `configs/` após a migração.
- Criar um loader de YAML que receba uma pasta de experimento e uma função de
  composição/normalização dos cinco arquivos daquela pasta.
- Validar campos obrigatórios, tipos, caminhos, target, métodos, métricas e
  parâmetros do grid; produzir mensagens de erro com o caminho do campo.
- Corrigir o YAML do experimento Iris para `data/raw/iris.csv` e incluir `task`,
  `selection` e `outputs`.
- Fazer cada `models.yaml`, `preprocessing.yaml`, `metrics.yaml` e
  `cross_validation.yaml` conter as escolhas daquele experimento, sem apontar
  para catálogos compartilhados.
- Criar um entrypoint real, por exemplo
  `python -m ml_playground.experiments.cli --experiment experiments/iris_baseline`,
  e um script equivalente em `pyproject.toml` se isso melhorar a ergonomia.
- Adicionar `--all` para executar todas as subpastas válidas de `experiments/`
  em sequência, preservando o nome de cada experimento nos relatórios.
- Garantir que o comando funcione a partir da raiz e não dependa do diretório
  corrente para localizar dados, código ou pastas de experimento.

Critério de aceite: um comando, sem código Python auxiliar, carrega uma pasta de
experimento, compõe os cinco YAMLs, expande o grid e inicia uma execução
identificável. `--all` executa mais de uma pasta sem misturar suas configurações.

### Fase 2 — Registry mínimo e pipeline único (P0)

- Adicionar `DecisionTreeClassifier` ao registry.
- Padronizar `logistic_regression`, preservando o alias de migração.
- Definir parâmetros válidos e grids pequenos, executáveis por padrão, para os
  quatro modelos.
- Fazer o executor montar um único `sklearn.pipeline.Pipeline` com pré-
  processamento e estimador final; cada fold deve receber uma instância nova.
- Usar `ColumnTransformer` para separar colunas numéricas e categóricas, com
  `handle_unknown` seguro no one-hot encoding.
- Alinhar cada item dos `preprocessing.yaml` ao registry ou removê-lo do YAML do
  experimento; nenhuma opção declarada pode resultar em `None` silencioso.
- Salvar o pipeline completo quando `save_model` estiver habilitado, incluindo
  config, versão, seed e métricas no metadata.

Critério de aceite: o modelo persistido consegue prever a partir das mesmas
colunas brutas usadas na entrada, sem reconstruir manualmente o pré-processamento.

### Fase 3 — Execução e avaliação robustas (P0)

- Implementar explicitamente `holdout`, `kfold`, `stratified_kfold` e
  `repeated_kfold`, escolhendo o splitter pelo nome da config.
- Aplicar estratificação automaticamente em classificação quando suportada e
  rejeitar combinações impossíveis antes do treinamento.
- Registrar, por combinação e por fold, tempo, tamanho das partições, seed,
  predições, scores/probabilidades e métricas.
- Definir métricas com comportamento explícito para binário e multiclasses
  (`macro`, `weighted` ou configuração equivalente). Usar todas as colunas de
  probabilidade no ROC AUC multiclass e tratar SVM com `decision_function` ou
  probabilidades habilitadas quando a métrica exigir.
- Representar métricas não escalares, como matriz de confusão, em JSON/tabela
  própria; não tentar inseri-las como um único valor CSV.
- Calcular média e desvio-padrão das métricas de CV e conservar os valores de
  cada fold.
- Manter erros por combinação com status, tipo e mensagem, sem abortar o grid
  inteiro; ao final, falhar o comando se nenhuma combinação for executada com
  sucesso.

Critério de aceite: o experimento do Iris conclui com os quatro modelos, cinco
folds estratificados e métricas válidas, sem erro de métrica multiclasses.

### Fase 4 — Grid, seleção e persistência de resultados (P0)

- Corrigir o contrato de `run_grid()` e criar um identificador estável para cada
  combinação de modelo, hiperparâmetros e pré-processamento.
- Fazer o tracker criar um `run_id` único por execução e guardar a config efetiva
  dentro de `reports/<experiment_name>/`; não criar `runs/` solto na raiz.
- Definir uma regra de seleção: métrica principal, direção (`max`/`min`) e
  desempate por métrica secundária e variabilidade.
- Publicar, dentro da pasta do experimento, um manifesto por execução com
  status, config, versões, dataset, combinações e caminhos dos artefatos.
- Salvar o pipeline do melhor modelo em
  `models/<experiment_name>/<run_id>/` e registrar o caminho no manifesto e na
  tabela de resultados.

Critério de aceite: a execução informa qual combinação venceu e é possível
reproduzir o resultado usando o nome do experimento, o `run_id`, o manifesto e a
configuração efetiva.

### Fase 5 — Layout de `reports/` por experimento e visualizações (P0/P1)

Centralizar a escrita de artefatos em um writer de relatórios. Nenhum executor
deve escolher caminhos arbitrários. Para cada pasta
`experiments/<experiment_name>/`, criar e reutilizar uma pasta homônima em
`reports/<experiment_name>/`, com a mesma estrutura de saída:

```text
reports/
  iris_baseline/
    metrics/
      <run_id>_fold_metrics.csv
      <run_id>_summary.csv
    tables/
      <run_id>_model_comparison.csv
      <run_id>_errors.csv
      <run_id>_<model>_confusion_matrix.csv
      <run_id>_manifest.json
    figures/
      <run_id>_metric_comparison.png
      <run_id>_roc_curves.png
      <run_id>_<model>_confusion_matrix.png
      <run_id>_<model>_feature_importance.png
  iris_without_scaling/
    metrics/
    tables/
    figures/
```

Regras:

- `metrics/` contém somente valores escalares e agregações de folds;
- `tables/` contém tabelas, matrizes, erros e manifesto;
- `figures/` contém somente imagens geradas, quando habilitadas;
- cada experimento possui exclusivamente a sua pasta em `reports/`; resultados
  de experimentos diferentes nunca podem ser misturados;
- todos os arquivos carregam o mesmo `experiment_name` pelo caminho e o mesmo
  `run_id` pelo nome, nunca sobrescrevendo outra execução;
- uma nova execução reutiliza a pasta do experimento, mas cria novos arquivos
  identificados por `run_id`;
- diretórios são criados sob demanda e o writer retorna os caminhos gerados;
- saídas não usadas por um modelo, como importância de features para KNN, devem
  ser marcadas como indisponíveis ou omitidas de forma documentada.

Critério de aceite: ao executar `experiments/iris_baseline/`, os artefatos são
criados somente em `reports/iris_baseline/metrics`,
`reports/iris_baseline/tables` e `reports/iris_baseline/figures`, com nomes
correlacionáveis ao manifesto. Ao executar um segundo experimento, ele recebe
uma pasta própria em `reports/` e nenhum CSV contém estruturas Python ilegíveis.

### Fase 6 — Logs, testes e documentação (P0)

- Integrar `setup_logger`, `log_summary` e `Timer` somente no script real de
  execução.
- Fazer `logs/<script>.log` sobrescrever a execução anterior e conter apenas
  warnings/erros durante o processamento, um resumo final e o tempo na última
  linha. Testes não devem criar logs persistentes.
- Adicionar testes para loader/composição/validação de YAML, aliases e árvore
  de decisão, splitters, métricas multiclasses, pipeline completo, seleção do
  melhor modelo, layout de relatórios, manifesto e rerun sem colisão de nomes.
- Adicionar um teste de fumaça que execute uma pasta do Iris e outra variante
  usando diretórios temporários para `experiments`, `reports`, `models` e logs;
  verificar que as duas árvores de relatório ficam separadas.
- Atualizar `README.md` com o comando de execução, o contrato das pastas de
  experimento, os modelos do MVP, o layout de `reports/` e a interpretação do
  relatório.
- Corrigir anotações de tipo, mensagens e encoding para que a API pública seja
  consistente.

Critério de aceite: a suíte passa em ambiente limpo e o smoke test reproduz o
fluxo documentado sem depender de notebooks.

## 5. Definição de pronto do MVP

O MVP estará pronto quando todos os itens abaixo forem verdadeiros:

- `experiments/iris_baseline/` executa a partir da raiz com um único comando;
- todos os YAMLs necessários ficam dentro da pasta do experimento, sem depender
  de arquivos soltos em `configs/`;
- KNN, regressão logística, SVM e árvore de decisão estão disponíveis no
  registry e no grid padrão;
- o mesmo pré-processamento é ajustado dentro de cada fold e fica incorporado
  ao pipeline salvo;
- holdout e validação cruzada estratificada produzem métricas por fold e resumo;
- problemas binários e multiclasses são suportados sem falhas silenciosas;
- combinações que falham aparecem em relatório com erro e não interrompem as
  demais;
- o melhor modelo é selecionado por regra configurável;
- métricas, tabelas, figuras e manifesto são alocados corretamente em
  `reports/<experiment_name>/metrics`,
  `reports/<experiment_name>/tables` e
  `reports/<experiment_name>/figures`;
- a última linha do log contém o tempo de execução e o resumo identifica
  processados, concluídos, falhos e ignorados;
- testes unitários e um teste de fumaça cobrem o caminho completo.

## 6. Fora do MVP

Não bloquear a primeira versão com otimização bayesiana, AutoML, MLflow/trackers
externos, ONNX, regressão, execução distribuída ou novos modelos além dos quatro
definidos. Esses itens só devem entrar depois que o contrato de configuração,
reprodutibilidade e persistência de relatórios estiver estável.
