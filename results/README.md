# Resultados

Cada experimento recebe uma pasta própria com previsões, métricas, configurações e gráficos. As execuções atuais são:

- `Teste1`: comparação principal cronológica 50/50.
- `Teste2`: sensibilidade 70/30.
- `Teste3`: inclusão de INGARCH Poisson e binomial negativa no corte 50/50, com diagnósticos por origem.
- `Teste4` e `Teste4Semente123`: sensibilidade 60/40 com sementes 42 e 123.

O arquivo [`sensibilidade_janelas.csv`](sensibilidade_janelas.csv) reúne as métricas e as diferenças pareadas contra o baseline em todas essas execuções.

Os cortes compartilham parte do período de teste e são sensibilidades, não replicações independentes. O código evita sobrescrever uma pasta que já contém resultados; use outro nome para cada execução. Artefatos de versões anteriores foram preservados em `legacy_optimizers/` e `legacy_hybrid/`, e não devem ser misturados aos novos.
