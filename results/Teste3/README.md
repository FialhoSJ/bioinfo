# Teste3 — INGARCH Poisson e binomial negativa

Comparação primária 50/50, com os mesmos 112 alvos de desenvolvimento até abril de 2017 e 113 alvos de teste de maio de 2017 a setembro de 2026 do `Teste1`. Foram incluídos INGARCH(1,1) com intercepto harmônico sazonal, distribuição condicional Poisson ou NB2. Não foram usadas variáveis de clima.

O INGARCH-NB obteve MAE 27,66, RMSE 36,16 e deviance de Poisson 12,63; o INGARCH-Poisson, 27,50, 36,43 e 12,75. O baseline do último mês obteve 27,07/37,18 e deviance 13,80. Os intervalos pareados de MAE dos dois INGARCH contra o baseline incluem zero.

Os 113 ajustes externos de cada modelo convergiram. A dispersão de Pearson dos resíduos preditivos foi 12,95 no Poisson e 0,99 no NB2. A soma média dos coeficientes do INGARCH-NB foi 0,916 (faixa 0,886–0,949); a dispersão NB2 média, 0,151 (faixa 0,142–0,165). A ACF dos resíduos de Pearson no lag 12 foi 0,11 e o p do Ljung–Box até lag 12 foi 0,65. Esses diagnósticos são descritivos e estão detalhados por origem em `diagnosticos_ingarch_por_origem.csv` e resumidos em `resumo_diagnosticos_ingarch.csv`.

Os arquivos restantes trazem previsões, métricas, configurações, validação interna, diferenças pareadas e gráficos. Consulte [`docs/revisao_e_protocolo.md`](../../docs/revisao_e_protocolo.md) para a interpretação completa.
