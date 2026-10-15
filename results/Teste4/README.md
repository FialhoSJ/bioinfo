# Teste4 — sensibilidade 60/40, semente 42

Divisão cronológica com 135 alvos de desenvolvimento até março de 2019 e 90 alvos de teste de abril de 2019 a setembro de 2026. A validação interna manteve três dobras temporais de 12 meses. Nenhuma variável climática foi usada.

SVR obteve MAE 21,92 e RMSE 30,00; o baseline do último mês, 23,72 e 31,76; SARIMA, 25,02 e 31,69. A diferença pareada de MAE SVR menos baseline foi -1,80 caso (IC95% [-5,06; 1,65]), incluindo zero.

Este corte se sobrepõe aos períodos dos `Teste1` e `Teste2`, portanto serve como análise de sensibilidade ao histórico e não como replicação independente. Os arquivos incluem diagnósticos por origem dos modelos INGARCH.
