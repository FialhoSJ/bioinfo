# Teste2 — sensibilidade 70/30

Execução da divisão cronológica com 157 alvos de desenvolvimento até janeiro de 2021 e 68 alvos de teste, de fevereiro de 2021 a setembro de 2026. Foram usadas três dobras temporais internas de 12 meses. O CSV usado está em `~/Downloads/`; consulte `resumo_execucao.json` para metadados.

ETS liderou MAE (23,18) e RMSE (31,04). SVR ficou com MAE 24,09 e RMSE 32,45; o baseline do último mês, 24,15 e 32,78. A diferença pareada de MAE SVR menos baseline foi -0,06 caso (IC95% [-3,59; 3,94]), sem evidência conclusiva de diferença.

Esta divisão é uma análise de sensibilidade. Seu período de teste se sobrepõe ao `Teste1` e não deve ser tratado como uma replicação independente. Os outros arquivos contêm previsões, métricas, configurações selecionadas, validação interna e gráficos. A análise consolidada está em [`docs/revisao_e_protocolo.md`](../../docs/revisao_e_protocolo.md).
