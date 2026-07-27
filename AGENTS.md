## Leis do Projeto

1. **Lei da Delegação Nua.** Toda função precisa justificar existência. Se o corpo é apenas delegar chamada idêntica a uma lib (com ou sem try/except), exclua. Só sobrevivem funções que **orquestram múltiplos passos** ou **resolvem formato/estratégia em tempo de execução**.

2. **Lei do Valor Zero.** Não embrulhe lib em função com nome diferente. Quem usa o código sabe chamar sklearn, polars, numpy direto. Adaptadores que só convertem tipo sem transformar lógica também são proibidos.

3. **Lei dos Logs.** Arquivo em `logs/<script>.log`, sobrescreve execução anterior, só erros/warnings, summary ao final, tempo de execução na última linha. Logs só existem para scripts reais — testes não geram logs persistentes.

4. **Lei da Raiz Limpa.** Planos de implementação ficam em `docs/`. Código executável fica em `src/ml_playground/`. Nada de planos, specs ou diagramas na raiz do projeto.
