# Teste4Semente123 — sensibilidade 60/40, semente 123

Repetição do corte `Teste4` (135 alvos de desenvolvimento e 90 de teste, abril de 2019 a setembro de 2026) com semente 123. SVR, SARIMA e os INGARCH são determinísticos aqui e reproduziram as mesmas previsões. Random Forest variou de MAE/RMSE 23,79/31,76 com semente 42 para 23,95/32,17 com semente 123. O baseline do último mês não muda com a semente.

Esta repetição mede a sensibilidade dos métodos estocásticos; não constitui novo período independente de teste. Nenhuma variável climática foi usada. Consulte [`docs/revisao_e_protocolo.md`](../../docs/revisao_e_protocolo.md) para a comparação entre cortes.
