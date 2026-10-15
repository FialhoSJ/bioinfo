# Próximos passos da pesquisa

Neste documento eu registro a sequência que vou seguir para avançar a comparação de previsão mensal de malária em Breves-PA. Eu mantenho o foco em prever contagens usando somente informações que estariam disponíveis na origem da previsão. Os experimentos novos ficam em pastas próprias para que eu possa revisar métodos e configurações sem sobrescrever resultados anteriores.

## 1. Consolidar o resultado atual

Eu trato `Teste1` como a referência inicial: divisão cronológica de 112 alvos para desenvolvimento e 113 para teste, com previsões mensais de um passo à frente. O SVR foi o melhor pontual em MAE e RMSE; a diferença pareada contra o baseline do último mês não foi conclusiva. O SARIMA teve RMSE quase igual e menor deviance de Poisson.

Antes de alterar o protocolo, eu vou preservar os arquivos atuais e registrar que:

- o SVR teve MAE 24,71 e RMSE 35,31;
- o baseline do último mês teve MAE 27,07 e RMSE 37,18;
- o intervalo de 95% da diferença pareada de MAE SVR menos baseline foi de -5,82 a 0,99 casos, incluindo zero;
- o SVR subestimou em média 6,44 casos;
- o teste não usou variáveis climáticas.

Eu uso essa leitura para orientar as próximas análises, sem escolher um método apenas por um pequeno ganho pontual. A revisão de [Zinszer et al. (2012)](https://doi.org/10.1136/bmjopen-2012-001992) recomenda avaliar métodos diferentes na mesma série com métricas comuns; por isso, eu mantenho os mesmos meses e baselines nas próximas comparações.

## 2. Confirmar qualidade e significado dos dados

Antes de adicionar novos modelos, eu vou revisar com a fonte e documentar:

1. **Diferença entre linhas e contagens — conferida internamente:** as 30.249 linhas válidas de Breves-PA têm `Número de registros = 1`. Treze linhas pertencem a outubro de 2026, mês parcial excluído; por isso, a série até setembro soma 30.236 notificações.
2. **Completude mensal — conferida no CSV:** a série mensal não tem lacunas nem contagens zero. A ausência de linhas em outros recortes ainda precisa ser confirmada com a fonte, pois não permite distinguir ausência real de casos de falha de vigilância ou extração.
3. **Definição temporal:** o alvo usa data de notificação. Ainda falta obter informação da fonte sobre atraso de notificação e revisões retrospectivas para saber quais valores estariam disponíveis em cada origem de previsão.
4. **Unidade do alvo:** sem denominador populacional, a saída continua descrita como contagem de notificações, não incidência ou risco.
5. **Data parcial — aplicada:** outubro de 2026 foi excluído porque o CSV termina em 5 de outubro.

Essa etapa importa porque zero observado, dado ausente e notificação atrasada têm significados diferentes. [Briët et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/) relatam ambiguidades entre zeros e dados ausentes em registros de vigilância; eu uso esse artigo como motivo para validar a regra de preenchimento antes de interpretar lacunas.

## 3. Executar análises de sensibilidade

Eu mudei uma decisão de cada vez e guardei cada saída separadamente. As divisões abaixo compartilham meses de teste e são análises de sensibilidade, não replicações independentes.

### Teste2 — proporção de desenvolvimento 70/30

Usei 70% dos alvos iniciais para desenvolvimento e 30% finais para teste, como análise de sensibilidade ao tamanho do histórico. Não escolhi a fração pelo menor erro; comparei ranking e viés entre janelas.

```bash
./run.sh Teste2 --train-fraction 0.7
```

O `Teste2` terminou com 157 alvos de desenvolvimento até janeiro de 2021 e 68 meses de teste. ETS liderou MAE (23,18) e RMSE (31,04); SVR e último mês ficaram praticamente empatados em MAE (24,09 e 24,15). A diferença pareada de MAE entre SVR e baseline foi -0,06 caso, com IC95% de -3,59 a 3,94.

### Teste3 — foco em previsão de contagem

Comparei baselines, Poisson, binomial negativa e INGARCH sazonal Poisson/NB na divisão primária 50/50, com a mesma validação temporal interna. O INGARCH-NB obteve MAE 27,66, RMSE 36,16 e deviance de Poisson 12,63; o INGARCH-Poisson obteve 27,50, 36,43 e 12,75. Ambos ficaram próximos do baseline, sem diferença conclusiva de MAE.

### Teste4 — robustez do SVR

Avaliei o corte 60/40 (135 alvos de desenvolvimento até março de 2019; 90 meses de teste) com as sementes 42 e 123, mantendo o baseline do último mês e o SARIMA como referências. SVR teve MAE 21,92 e RMSE 30,00 nas duas sementes; como é determinístico neste código, a semente não altera suas previsões. O baseline teve MAE 23,72/RMSE 31,76 e SARIMA 25,02/31,69. A repetição de semente alterou Random Forest de MAE 23,79/RMSE 31,76 para 23,95/32,17, mas não mudou a conclusão sobre os métodos de referência.

Com os três cortes, SVR liderou MAE em 50/50 e 60/40, enquanto ETS liderou 70/30. Os intervalos pareados SVR-baseline incluíram zero nos três cortes. Os testes se sobrepõem no tempo, então não os interpreto como evidência independente nem como confirmação de superioridade do SVR.

Eu não vou apresentar várias divisões sobrepostas como testes independentes. A avaliação fora da amostra precisa permanecer temporal e usar medidas padronizadas, como recomenda a revisão mais recente de [Mategula et al. (2025)](https://doi.org/10.1371/journal.pgph.0003751).

## 4. Priorizar métodos de contagem com dependência temporal

Além dos GLM Poisson e binomial negativa com defasagens, implementei INGARCH(1,1) com intercepto sazonal harmônico, em versões Poisson e binomial negativa. A média condicional usa a contagem anterior e a média condicional anterior; a soma dos coeficientes dinâmicos é restringida a menos de 0,98. A estimação por máxima verossimilhança foi refeita em cada origem do teste.

Nas 113 origens do `Teste3`, todos os ajustes convergiram. A dispersão de Pearson dos resíduos preditivos foi 12,95 para INGARCH-Poisson e 0,99 para INGARCH-NB. O INGARCH-NB teve ACF de resíduo de 0,11 no lag 12 e p=0,65 no Ljung–Box até lag 12; são diagnósticos descritivos do teste, não prova de adequação. Seus coeficientes médios foram α=0,862 e β=0,054 (soma média 0,916; faixa 0,886–0,949), com dispersão NB2 média 0,151 (faixa 0,142–0,165). O Poisson ficou fortemente sobredisperso.

Os candidatos de contagem que seguem em aberto são:

1. **GLARMA Poisson/binomial negativa** ou **INAR**, para ampliar a comparação de dependência temporal nas contagens.
2. **GSARIMA/GARMA binomial negativa**, para combinar sazonalidade anual, dependência temporal e distribuição de contagem.
3. **Modelo Bayesiano dinâmico binomial negativo**, se eu precisar comunicar intervalos preditivos e incerteza além da previsão média.

[Briët et al. (2013)](https://doi.org/10.1371/journal.pone.0065761) desenvolveram GSARIMA/GARIMA para contagens de malária e motivam os dois primeiros grupos. [Diao et al. (2023)](https://doi.org/10.3390/ijerph20136303) comparam Poisson e binomial negativa com histórico de casos; eu uso esse estudo para justificar as duas distribuições, sem transferir para Breves o resultado obtido com clima e dados do Senegal.

Eu já revisei autocorrelação dos resíduos, dispersão condicional e estabilidade dos parâmetros nas previsões externas do INGARCH. A razão variância/média bruta é aproximadamente 147, mas a tendência e mudanças de regime podem inflá-la; ela segue como sinal para investigar, não como diagnóstico definitivo.

## 5. Adicionar clima somente quando houver dados alinhados

Quando eu obtiver chuva, temperatura e, se possível, umidade, vou criar uma nova etapa com covariáveis defasadas. Para cada variável, eu vou:

1. conferir cobertura, frequência e qualidade da fonte;
2. alinhar cada observação ao mês em que estaria disponível para a previsão;
3. selecionar defasagens usando somente o conjunto de desenvolvimento;
4. comparar o modelo com casos passados contra o mesmo modelo com casos e clima;
5. medir se o clima melhora o teste externo e se reduz erros nos picos.

[Sewe et al. (2017)](https://doi.org/10.1038/s41598-017-02560-z) motivam testar GAM/GAMBoost com informação ambiental: no estudo queniano, GAMBoost teve bom desempenho com clima por satélite e casos passados. Eu vou tratar esse resultado como hipótese para testar em Breves, não como garantia de ganho. Se eu obtiver população e intervenções com cobertura compatível, vou avaliar esses fatores em ablações próprias.

## 6. Adiar redes profundas e ampliar a comparação de ML com cautela

Eu vou manter Random Forest, boosting Poisson e SVR como referências de aprendizado de máquina. Com apenas 112 alvos de desenvolvimento, eu não vou priorizar LSTM/GRU. [Barboza et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9344950/) usaram agrupamentos de municípios do Amazonas e encontraram diferenças entre grupos e métodos; eu usaria esse trabalho como base para uma etapa futura com painel municipal, caso eu tenha séries comparáveis para outras localidades.

Se a amostra aumentar, eu vou considerar GAM com resposta Poisson/NB, GAMBoost e métodos de boosting com objetivo de contagem. Cada candidato vai usar a mesma validação temporal e um orçamento de ajuste comparável.

## 7. Manter otimizadores fora da comparação principal de previsores

Eu não vou comparar GA, PSO e BAT como se fossem modelos que preveem casos. Para a comparação principal, vou usar estimação por máxima verossimilhança ou Bayesiana nos modelos estatísticos e uma grade pequena, busca aleatória ou otimização bayesiana dentro da validação interna para os modelos de ML.

Eu só retomarei GA/PSO/BAT se formular uma pergunta secundária sobre estratégias de busca. Nesse caso, vou aplicá-los ao mesmo preditor, limitar igualmente o número de avaliações, repetir sementes e manter o teste externo sem participação na busca.

## 8. Padronizar o que eu vou reportar

Para cada método, eu vou guardar e revisar:

- MAE e RMSE em casos;
- MASE calculado apenas com o desenvolvimento;
- viés médio, com sinal indicando sub ou superestimação;
- sMAPE e deviance de Poisson como métricas complementares;
- diferenças pareadas contra o baseline do último mês, com intervalo em blocos temporais;
- gráficos de previsões e resíduos, com atenção aos picos e às mudanças de nível;
- configurações selecionadas sem consultar os alvos de teste.

Se o objetivo evoluir de previsão de contagem para emissão de alertas de surto, eu vou definir primeiro um limiar operacional com a equipe usuária. Depois compararei sensibilidade, falsos alertas e antecedência do aviso; não vou inferir qualidade de alerta apenas a partir do RMSE.

## Ordem de execução resumida

1. Preservar `Teste1` como comparação primária e tratar `Teste2`, `Teste4` e a repetição de semente como sensibilidades sobrepostas.
2. Confirmar com a fonte o atraso e a revisão retrospectiva das notificações; a checagem do CSV não resolve essa disponibilidade operacional.
3. Ampliar modelos temporais de contagem para GLARMA/INAR ou GSARIMA binomial negativa, mantendo a validação temporal e os diagnósticos.
4. Integrar clima ou outras covariáveis apenas se forem obtidas e alinhadas ao instante real da previsão; elas não entraram nos experimentos executados.
5. Confirmar conclusões com meses futuros ainda não examinados ou outra série municipal compatível.

Eu registro o protocolo, os artigos e os resultados de cada etapa em [`revisao_e_protocolo.md`](revisao_e_protocolo.md) e em suas pastas `results/TesteN/` correspondentes.

## Referências que orientam este roteiro

- Briët OJT, Vounatsou P, Gunawardena DM, Galappaththy GNL, Amerasinghe PH. (2008). Models for short term malaria prediction in Sri Lanka. *Malaria Journal*, 7:76. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/). Eu uso para justificar validação temporal fora da amostra, inclusão de métodos clássicos e checagem de zeros/dados ausentes.
- Briët OJT, Amerasinghe PH, Vounatsou P. (2013). Generalized Seasonal Autoregressive Integrated Moving Average Models for Count Data with Application to Malaria Time Series with Low Case Numbers. *PLOS ONE*, 8(6):e65761. [DOI](https://doi.org/10.1371/journal.pone.0065761). Eu uso para priorizar GSARIMA/GARMA com distribuição de contagem.
- Ferland R, Latour A, Oraichi D. (2006). Integer-Valued GARCH Process. *Journal of Time Series Analysis*, 27(6):923–942. [DOI](https://doi.org/10.1111/j.1467-9892.2006.00496.x). Eu uso para a estrutura INGARCH Poisson com média condicional dinâmica.
- Zhu F. (2011). A negative binomial integer-valued GARCH model. *Journal of Time Series Analysis*, 32(1):54–67. [DOI](https://doi.org/10.1111/j.1467-9892.2010.00684.x). Eu uso para estender a distribuição condicional à binomial negativa.
- Zinszer K, Verma AD, Charland K, et al. (2012). A scoping review of malaria forecasting: past work and future directions. *BMJ Open*, 2:e001992. [DOI](https://doi.org/10.1136/bmjopen-2012-001992). Eu uso para manter os métodos na mesma série e reportar métricas comuns.
- Diao O, Absil P-A, Diallo M. (2023). Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal. *International Journal of Environmental Research and Public Health*, 20(13):6303. [DOI](https://doi.org/10.3390/ijerph20136303). Eu uso para comparar regressões Poisson e binomial negativa com histórico de casos.
- Sewe MO, Tozan Y, Ahlm C, Rocklöv J. (2017). Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya. *Scientific Reports*, 7:2589. [DOI](https://doi.org/10.1038/s41598-017-02560-z). Eu uso como motivação para testar clima e GAMBoost quando houver covariáveis alinhadas.
- Barboza D, et al. (2022). Prediction of malaria using deep learning models: A case study on city clusters in the state of Amazonas, Brazil, from 2003 to 2018. *Revista da Sociedade Brasileira de Medicina Tropical*. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9344950/). Eu uso como referência regional para uma etapa futura com várias séries municipais, não como justificativa para iniciar com LSTM/GRU em Breves.
- Mategula D, et al. (2025). Advancing Early Warning Systems for Malaria: Progress, challenges, and future directions — A scoping review. *PLOS Global Public Health*, 5(5):e0003751. [DOI](https://doi.org/10.1371/journal.pgph.0003751). Eu uso para reforçar avaliação fora da amostra e diagnóstico de autocorrelação.
