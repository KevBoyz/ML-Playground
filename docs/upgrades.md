# Plano de upgrades — ML-Playground para treinamento prático

## 1. Decisão e objetivo

Este é o próximo ciclo depois do MVP em docs/FirstPlan.md e do Beta em
docs/Beta.md. Ele **não** reimplementa classificação, regressão ou
clusterização: os três fluxos, os experimentos autocontidos e os relatórios por
run_id já existem.

O objetivo agora é tornar o repositório confiável para transformar um estudo de
notebook em um experimento tabular repetível:

1. receber uma tabela já preparada e uma configuração explícita;
2. validar dados e custo antes de treinar;
3. escolher candidatos sem contaminar a avaliação final;
4. registrar dados, splits, ambiente e artefatos usados;
5. recuperar o campeão para pontuar novas linhas com validação de schema.

O escopo continua sendo ML tabular local. Não entram conversão automática de
notebooks, download de bases, engenharia de atributos de domínio, AutoML, deep
learning, execução distribuída ou inferência/serving web.

## 2. Diagnóstico consolidado

O baseline está saudável: em 05/08/2026 a suíte passou com **103 testes**. Os
quatro exemplos exercitam os três tipos de tarefa, geram modelo, métricas,
predições, tabelas, figuras e manifesto. Portanto, o problema não é falta de
runner; é a distância entre um demo e uma rotina de treinamento real.

| Evidência atual | Lacuna prática | Consequência a corrigir |
| --- | --- | --- |
| runner.py escolhe o vencedor pela mesma métrica de holdout/CV usada para reportar resultado e depois faz refit com todas as linhas. | Não existe teste final bloqueado nem nested CV. | A métrica do campeão é otimista e não deve ser tratada como performance final. |
| executor.py só oferece holdout/KFold e variantes estratificadas. | Não há split por grupo, data, backtest ou arquivo de teste externo. | Estudos como futebol podem vazar futuro ou a mesma entidade entre treino e teste. |
| Os exemplos atuais expandem 125 candidatos e 955 fits; o último --all levou 201,08 s mesmo em bases pequenas. | Grid é sequencial, sem orçamento, prévia, retomar execução ou política de paralelismo. | A escala fica imprevisível em dados reais. |
| predictions.csv identifica apenas modelo, fold e índice posicional. No Iris, 150 linhas originam 8.700 predições; variações do mesmo modelo não são distinguíveis. | Faltam IDs de candidato, repetição, papel do split e ID de negócio. | Não é possível fazer análise de erro ou auditoria por observação. |
| O manifesto normaliza a configuração, mas usa paths absolutos e não inclui fingerprint da base, schema, índices de split, commit ou ambiente. | A execução não pode ser verificada ou reproduzida em outra máquina. | Um run_id não é evidência suficiente de reprodutibilidade. |
| O artefato joblib não declara schema de entrada nem capacidade de inferência; a única recuperação documentada é chamada Python interna. | Não há pontuação em lote segura nem promoção explícita de modelo. | É fácil pontuar colunas erradas ou usar clusterizador não indutivo como se previsse novas linhas. |
| PipelineCache não é usado e sua chave contém apenas configuração, sem dados, target ou fold. | Integrá-lo como está pode reutilizar transformador ajustado em outro split. | Há risco de resultado stale e leakage; ele não deve ser ativado. |
| validate_data() e build_column_transformer() não participam do fluxo; o loader não aceita opções de leitura. | Não há profile pré-fit, contrato de tipos, ID, separador, encoding ou sheet configuráveis. | CSVs/Excel reais e colunas mistas exigem código manual antes do experimento. |
| README.md, USAGE.md e os planos históricos têm referências e status divergentes. | O usuário não tem matriz única do que funciona hoje. | Views ou dependências opcionais podem ser configuradas com expectativa errada. |

Os upgrades abaixo preservam a fronteira correta do Beta: o pacote coordena um
experimento; a receita de negócio que produz a tabela preparada continua
versionada fora do core e declarada pelo usuário.

## 3. Princípios de desenho

- **Avaliação antes de variedade.** Nenhum modelo ou view novo deve esconder a
  ausência de teste final, split adequado ou proveniência.
- **Configuração explícita e congelada.** Defaults são documentados; o
  manifesto contém a configuração efetiva. Não haverá herança silenciosa entre
  pastas de experimento.
- **Colunas com papéis.** ID, metadados, grupo e tempo nunca viram feature por
  acidente, mas acompanham predições e splits quando solicitados.
- **Artefato portátil.** Paths armazenados no run são relativos à raiz do
  projeto/artifact store; hashes e versões permitem verificar o conteúdo em
  outra máquina.
- **Custo sob controle.** Toda busca informa quantos fits executará e respeita
  orçamento, paralelismo e política de erro explícitos.
- **Sem delegação nua.** Código novo em src/ml_playground/ só deve orquestrar
  ciclo de vida, resolver estratégia ou validar contratos. Não criar wrappers
  que apenas renomeiem APIs de sklearn, polars ou numpy.
- **Logs e raiz limpa.** Scripts reais continuam escrevendo somente
  warnings/erros, resumo e tempo final em logs/<script>.log; planos ficam em
  docs/ e testes não deixam logs persistentes.

## 4. Fluxo-alvo de uma execução

~~~text
tabela preparada + receita externa
  -> validate/profile/fingerprint
  -> plano de splits congelado
  -> busca somente no conjunto de desenvolvimento
  -> seleção documentada
  -> avaliação única no teste bloqueado (ou estimativa nested CV)
  -> refit de deploy, se solicitado
  -> run store + relatório + model card
  -> predict em novas linhas com schema validado
~~~

Para classificação e regressão, as métricas precisam declarar sua origem:
development_cv, validation, outer_cv ou final_test. Um teste final nunca
participa de escolha de parâmetro, threshold, preprocessing ou modelo. Para
clusterização, a seleção inicial continua interna ao dataset declarado, com
restrições e estabilidade por reamostragem; CV/teste externo só entram quando a
semântica da segmentação os justificar e forem configurados explicitamente.

## 5. Contrato-alvo e migração

Antes de ampliar YAMLs, projetar uma versão de contrato com migração explícita
dos campos atuais. O formato final pode ajustar nomes, mas deve representar os
seguintes conceitos:

~~~yaml
data:
  path: data/processed/churn.parquet
  read_options: {}          # separador, encoding, sheet, null_values
  test:
    path: data/processed/churn_test.parquet
    read_options: {}
  target: churned           # apenas supervisionado
  id_column: customer_id    # preservado nos artefatos; nunca é feature
  metadata_columns: [region]
  group_column: household_id
  time_column: event_at
  features:
    numeric: [age, monthly_fee]
    categorical: [plan]
  schema:
    mode: strict            # regra para coluna ausente, extra e dtype

evaluation:
  protocol: train_validation_test  # ou nested_cv
  splitter: {name: stratified_group_kfold, n_splits: 5}
  final_test: {source: path}       # path ou split

selection:
  primary_metric: f1_macro
  direction: maximize
  tie_breakers: [metric_std, candidate_id]
  baseline_candidate: dummy_classifier  # opcional; deve ser candidato válido

provenance:
  recipe_ref: recipes/churn_features.py  # referência; nunca executada pelo core
  recipe_revision: <git-revision-ou-hash>
  source_description: snapshot preparado de origem licenciada

search:
  strategy: grid
  max_candidates: 40

execution:
  n_jobs: 4
  max_wall_time_seconds: 3600
  on_candidate_error: continue
~~~

Os campos data.path, data.features, cross_validation e selection atuais devem
continuar legíveis durante uma janela de compatibilidade. O loader emite aviso
de migração e sempre grava a versão do contrato, a configuração resolvida e a
referência não executável da receita; não deve adivinhar grupo, tempo, ID ou
features.

## 6. Fases de implementação

### Fase 0 — consolidar o estado entregue (P0)

**Resultado:** uma fonte de verdade descreve o que já funciona e o ponto de
partida deixa de carregar infraestrutura enganosa.

- Criar página curta de capacidades/status: tarefa, modelo, splitter, métrica,
  view e comando realmente implementados; apontar README.md e USAGE.md para ela.
- Corrigir referências obsoletas, nomes divergentes de exemplos e documentação
  de dependências opcionais/preflight.
- Registrar artefatos atuais como demos históricos e regenerá-los após cada
  mudança de contrato; manifesto deve validar que todo caminho listado existe e
  tem checksum.
- Remover PipelineCache órfão ou substituí-lo somente na Fase 3. Não implementar
  a pendência do primeiro plano ligando o cache atual ao executor.
- Revisar módulos e APIs não integrados; manter apenas abstrações que
  orquestram um fluxo real, conforme AGENTS.md.

**Aceite:** um leitor distingue funcionalidades entregues, planejadas e não
suportadas; nenhum exemplo ou manifesto aponta para artefato inexistente; a
suíte e os exemplos de fumaça continuam verdes.

### Fase 1 — contrato de dados, preflight e proveniência (P0)

**Resultado:** uma execução inválida falha antes de fit, e uma execução válida
deixa evidência verificável da tabela usada.

- Integrar etapa única de preflight que lê a tabela uma vez e valida colunas,
  target, tipos, cardinalidade, nulos, duplicatas, classes raras e coerência
  entre tarefa, modelo, métrica e preprocessing.
- Suportar read_options por formato: encoding/separador/quote para CSV; sheet
  para planilha; nulls e opções suportadas pelo leitor. Não esconder heurísticas
  de leitura.
- Introduzir papéis de coluna: features por grupo, id_column, metadados, grupo e
  tempo. IDs/metadados são excluídos da matriz de treino e preservados em
  splits, predições, resíduos e labels.
- Validar id_column como chave não nula e única no escopo de cada fonte. Sem um
  ID de negócio, emitir source_row_id associado ao fingerprint e declarar que
  ele não sobrevive à reordenação; protocolos de grupo/tempo, teste externo,
  resume ou auditoria de predições podem exigir ID real e falhar sem ele.
- Gerar data_profile leve por padrão e detalhado de forma opt-in: shape, schema
  observado, missingness, duplicidade, cardinalidade, distribuição do target e
  avisos. Gatilhos configuráveis podem transformar avisos em falha; limpeza
  continua explícita no pipeline.
- Calcular fingerprint criptográfico do arquivo/dados, hash da configuração
  efetiva e assinatura de schema; registrar versão do pacote, Python,
  dependências/uv.lock, commit e estado Git, comando, timezone e seeds.
- Registrar provenance.recipe_ref, revisão e descrição da fonte sem executar a
  receita de preparo. A referência liga o experimento ao estudo de notebook ou
  script que gerou a tabela, preservando a fronteira do core.
- Quando houver data.test.path, validar antes do treino schema, target, ID e
  compatibilidade com data.path; fingerprint e profile do teste são distintos
  dos dados de desenvolvimento.
- Fazer preflight de capacidades: dependência opcional instalada, modelo aceita
  a tarefa/dados, métricas exigem score/probabilidade e transforms são
  compatíveis com tarefa e colunas. Centralizar política de seed para que o
  usuário não a repita em todo parâmetro de modelo.
- Expor ml-playground validate --experiment ... e --dry-run, ambos sem treino.
  O dry-run mostra schema, warnings, candidatos, fits previstos e
  incompatibilidades.

**Aceite:** alterar arquivo, schema, coluna de ID ou YAML altera a identidade
do run; CSV Latin-1/planilha com opção declarada é lido sem código auxiliar;
erros de coluna/tipo/capacidade aparecem antes do primeiro candidato.

### Fase 2 — protocolo de avaliação honesto (P0)

**Resultado:** seleção, avaliação final e refit são etapas distintas e
auditáveis.

- Para classificação e regressão, substituir o caminho único de holdout/CV por
  protocolos explícitos: train/validation/test, CV de desenvolvimento com teste
  externo bloqueado e nested CV quando não houver teste separado.
- Persistir plano de splits com versão, seed, papel de cada linha e IDs
  estáveis. A mesma tabela reordenada não pode mudar ID de negócio nem permitir
  reutilizar run/cache indevidamente.
- Acrescentar GroupKFold, StratifiedGroupKFold, holdout por grupo,
  TimeSeriesSplit/corte temporal e backtest tabular; validar que grupo e futuro
  não aparecem simultaneamente em treino e avaliação.
- Permitir estratificação por faixas declaradas em regressão, quando apropriado
  ao problema, sem inferir bins ou semântica de domínio automaticamente.
- Tornar métricas parametrizadas em vez de ambíguas: média, classe positiva,
  zero_division, labels esperados e direção. Validar probabilidade/score antes
  de ROC AUC ou log loss.
- Preservar o contrato de seleção: métrica primária escalar de desenvolvimento,
  direção, desempate determinístico, baseline opcional e tratamento explícito
  de candidato falho ou métrica indisponível. O que seleciona o campeão não é
  automaticamente o que se divulga como resultado final.
- Produzir representação OOF canônica: cada observação aparece uma vez por
  split/repeat/candidato identificável. Views de matriz, ROC e resíduos devem
  dizer se usam holdout, outer-CV ou OOF, nunca concatenar repeats como se
  fossem observações independentes.
- Depois da escolha, executar avaliação única no teste bloqueado e só então
  criar, se solicitado, artefato de deploy refitado em treino+validação. O
  manifesto separa selected_model, final_test e deployment_refit.
- Para clusterização, formalizar contrato próprio: métricas internas, limites de
  ruído/tamanho, estabilidade por reamostragem e perfil de clusters. Não exigir
  final_test/nested CV por padrão, nem chamar uma partição temporal de validação
  sem uma hipótese declarada de estabilidade.

**Aceite:** teste automatizado prova que teste final não influencia a seleção;
outro prova isolamento de grupo/tempo. Relatórios exibem métricas de
desenvolvimento e teste final em colunas distintas, com seus splits.

### Fase 3 — execução, busca e cache seguros (P0)

**Resultado:** o usuário sabe o custo antes de iniciar e pode executar buscas
maiores sem perder estado ou vazar dados.

- Criar candidate_id determinístico a partir de modelo, parâmetros,
  preprocessing, fingerprint da base e plano de splits. Criar também trial_id,
  repeat, fold e split_role para cada resultado.
- Fazer dry-run calcular candidatos, fits, memória estimada quando possível e
  avisar sobre produto cartesiano. Suportar limites de candidatos, tempo total,
  timeout por candidato, política continue/fail_fast e cancelamento limpo.
- Persistir estado de trial de forma incremental e retomar apenas combinações
  compatíveis com mesma configuração, dados, código, splits e ambiente/lock de
  dependências. Mudanças em qualquer uma dessas identidades invalidam a
  retomada.
- Adicionar paralelismo deliberado por candidato/fold e política contra
  paralelismo aninhado (execution.n_jobs versus model.n_jobs), mantendo ordem e
  seeds determinísticas.
- Incluir busca aleatória com orçamento depois da grade limitada; otimização
  bayesiana só é avaliada quando protocolos de avaliação e resume estiverem
  estáveis.
- Se cache for necessário, usar mecanismo seguro do pipeline ou cache de
  transformações com chave que inclua fingerprint dos dados, índices de treino,
  versão do transformador e configuração. Nunca armazenar pré-processador
  fitted identificando-o apenas pelo YAML.
- Não reter pipelines fitted de todos os candidatos em memória. Conservar
  resultados serializáveis, artefatos necessários e refazer apenas campeão
  conforme protocolo definido.

**Aceite:** interromper e retomar run não repete trials concluídos; mudança de
dados causa cache miss; grid grande respeita orçamento; teste de CV prova que
nenhuma transformação de fold é reutilizada em outro.

### Fase 4 — run store, relatórios e comparação operacional (P1)

**Resultado:** um run pode ser auditado por máquina e entendido por uma pessoa
sem abrir CSVs ambíguos.

- Definir run store portátil e imutável por experimento/run, preservando
  organização atual de reports/ durante migração quando ela não quebrar links.
  O manifesto vira contrato versionado, com checksums e paths relativos.
- Escrever tabelas de trial com candidate_id, parâmetros normalizados,
  direção/rank, duração, recursos quando disponíveis, baseline, delta e
  incerteza. Não converter métrica ausente em zero.
- Trocar predições genéricas por tabelas com candidate_id, trial_id, fold,
  repeat, split_role, ID de negócio, metadados autorizados, observado, predito,
  score/probabilidade e versão de schema. Salvar apenas campeão/OOF/teste por
  padrão; manter todos os candidatos como opção explícita e usar Parquet para
  volume maior.
- Gerar model_card.md (e, se necessário, HTML) com finalidade, dados,
  protocolo, campeão, baseline, métricas finais, limites, artefatos e comando
  de reprodução. Não incluir dados crus por padrão.
- Adicionar comandos show, compare e reproduce, que comparem somente runs com
  tarefa, métrica, dataset e protocolo compatíveis. comparison.py atual não
  deve ser promovido antes dessa semântica existir.
- Manter reports/, models/ e logs/ ignorados no Git; configurações, recipes,
  fingerprints e modelo de relatório são a fonte versionada. Artifact root
  externo/configurável é extensão posterior, sem exigir tracker externo agora.

**Aceite:** toda linha de predição e métrica é rastreável a candidato e split;
manifesto pode ser verificado sem paths de Windows; relatório explica por que o
campeão venceu e onde estão seus dados de avaliação.

### Fase 5 — inferência segura e criação rápida de estudos (P1)

**Resultado:** o campeão é utilizável depois do treino e iniciar estudo não
exige copiar código de notebook.

- Salvar, junto ao pipeline, assinatura de entrada: features, ordem, dtype,
  categorias/política de desconhecidos, transformações resultantes, fingerprint
  de treino, protocolo, capacidade predict/predict_proba e limitações de
  clusterizadores.
- Implementar ml-playground predict --model ... --input ... --output .... Ele
  orquestra carga, preflight de schema, preservação de ID/metadados e escrita de
  previsões; não é wrapper renomeado de pipeline.predict.
- Rejeitar ou relatar com precisão colunas ausentes, extras e dtypes
  incompatíveis conforme política do modelo. Para DBSCAN/agglomerativo,
  declarar no card quando não há atribuição de novas linhas.
- Criar ml-playground init com templates autocontidos e pequenos por arquétipo,
  além de validate, dry-run e run como caminho documentado. Templates aceleram
  início sem introduzir herança implícita entre experimentos.
- Criar aliases/promover modelo somente por comando explícito (candidate,
  champion, por exemplo), com registro de quem e qual run foi escolhido.

**Aceite:** tabela nova válida é pontuada sem código Python auxiliar e preserva
ID; tabela incompatível falha antes da predição; estudo novo nasce de template,
valida e executa com dados preparados e YAMLs.

### Fase 6 — diagnóstico e cobertura analítica orientados a decisão (P2)

**Resultado:** diagnósticos ajudam a decidir, sem fingir automatizar
conhecimento de domínio.

- Evoluir preprocessing para papéis/colunas explícitos, tratamento por tarefa,
  lineage de nomes após encoding/seleção/PCA e recipes versionadas. Target
  transform, imputação avançada ou polinômios entram apenas quando declarados e
  avaliados dentro do split.
- Priorizar classificação desbalanceada: balanced accuracy, PR AUC, métricas
  por classe, calibração e threshold/custo escolhido no desenvolvimento e
  congelado antes do teste final.
- Acrescentar importância por coeficiente/permutação quando válida; análises de
  erro por metadado autorizado; diagnósticos de regressão (resíduos,
  distribuição, QQ quando aplicável) com origem de split clara.
- Para clusterização, adicionar restrições de ruído/tamanho mínimo na seleção,
  estabilidade por reamostragem, perfil por feature original e projeção apenas
  como visualização. Silhouette sozinho não deve vencer se violar restrição de
  negócio declarada.
- Deixar tarefa estatística declarativa dos notebooks de carros/Série A como
  extensão separada: estatística e p-valor reportados, sem concluir
  automaticamente hipótese de negócio.

**Aceite:** cada view publicada tem dados de entrada, origem de split e motivo
de indisponibilidade; diagnóstico nunca duplica silenciosamente observações de
CV repetida nem aplica limpeza/transformação implícita.

## 7. Migração dos notebooks por arquétipo

| Arquétipo | Referências | Primeiro template e requisito de upgrade |
| --- | --- | --- |
| Classificação tabular mista | Titanic, Wine, Heart Disease, Heart Chains | Colunas numéricas/categóricas explícitas, ID/metadados, baseline, CV estratificada; depois PR/calibração se houver desbalanceamento. |
| Classificação cronológica | Apostas de futebol | Data de corte/backtest, features já preparadas fora do core, teste futuro bloqueado e proibição de leakage temporal. |
| Regressão/preparação | Vendas, Salary, California Houses, Life Expectation, Crazy Reg | Teste final, baseline, resíduos e schema de inferência; transformações de target/feature só declaradas. |
| Segmentação | Customer Segmentation, Credit Card Segmentation | Seleção com limites de ruído/tamanho, profile dos clusters e declaração de capacidade para novas linhas. |
| Exploração/estatística | Série A e Cars | Profile de dados primeiro; análises estatísticas declarativas somente na Fase 6. |

Cada template deve usar fixture pequena e dados sintéticos ou licenciados para
teste. O notebook permanece como referência de preparo/hipótese; o experimento
passa a representar a parte repetível de treino, avaliação e publicação.

## 8. Ordem de entrega e qualidade

1. Fazer Fase 0 e Fase 1 antes de mudar exemplos atuais. Na Fase 1, entregar
   também um vertical slice pequeno (Titanic misto ou futebol temporal) com
   template, recipe_ref, preflight e smoke test.
2. Só habilitar paralelismo, resume, random search ou cache depois da Fase 2,
   pois dependem de identidade estável de dados e splits.
3. Publicar novo run store/model card antes de criar comandos de comparação ou
   promoção.
4. Migrar um template de cada arquétipo e executar smoke tests em datasets
   pequenos antes de expandir views/modelos.

Além dos testes existentes, incluir testes de contrato para migração de YAML,
preflight, opção de leitura, fingerprint, seed e schema; testes anti-leakage de
preprocessing/cache/teste final/grupo/tempo; testes de determinismo,
timeout/resume e integridade de artefatos; e testes de predict com schema válido
e inválido. Os testes usam diretórios temporários e não criam logs persistentes.

## 9. Definição de pronto do núcleo prático

O núcleo estará pronto quando um usuário conseguir pegar uma tabela preparada de
um notebook, criar experimento por template, executar validate e dry-run,
treinar dentro de orçamento e obter tudo abaixo sem código auxiliar:

- avaliação de desenvolvimento separada de teste final, ou nested CV
  explicitamente marcado;
- split reproduzível por ID, grupo ou tempo quando declarado;
- campeão e predições rastreáveis a dados, candidato e split;
- manifesto portátil com configuração congelada, fingerprints, ambiente e
  checksums;
- artefato de deploy com schema validado e comando de pontuação em lote;
- relatório legível com baseline, ranking, métricas finais, limitações e comando
  de reprodução;
- regressão dos exemplos atuais coberta por smoke tests e conformidade com as
  leis de arquitetura, logs e organização do repositório.
