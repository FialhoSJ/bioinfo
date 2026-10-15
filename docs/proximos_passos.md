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

1. **Diferença entre linhas e contagens:** a extração atual tem 30.249 linhas válidas de Breves-PA e soma 30.236 registros. Eu vou verificar a origem dessa diferença e confirmar se `Número de registros` é o peso correto para agregar.
2. **Completude mensal:** a série não tem meses sem linhas nem meses com contagem zero. Eu vou confirmar se a ausência de linhas em outros recortes realmente representa ausência de casos ou se pode indicar falha de vigilância/extração.
3. **Definição temporal:** o alvo atual usa data de notificação. Eu vou obter informação sobre atraso de notificação e revisões retrospectivas para saber quais valores estariam disponíveis em cada origem de previsão.
4. **Unidade do alvo:** sem denominador populacional, eu vou chamar a saída de contagem de notificações, não de incidência ou risco.
5. **Data parcial:** eu vou continuar excluindo o último mês quando o CSV não cobrir o mês inteiro.

Essa etapa importa porque zero observado, dado ausente e notificação atrasada têm significados diferentes. [Briët et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/) relatam ambiguidades entre zeros e dados ausentes em registros de vigilância; eu uso esse artigo como motivo para validar a regra de preenchimento antes de interpretar lacunas.

## 3. Executar análises de sensibilidade

Eu vou mudar uma decisão de cada vez e guardar cada saída separadamente.

### Teste2 — proporção de desenvolvimento 70/30

Eu vou usar 70% dos alvos iniciais para desenvolvimento e 30% finais para teste, como análise de sensibilidade ao tamanho do histórico. Não vou escolher entre 50/50 e 70/30 pelo menor erro: vou verificar se o ranking e os vieses se mantêm quando a janela muda.

```bash
./run.sh Teste2 --train-fraction 0.7
```

### Teste3 — foco em previsão de contagem

Depois de revisar os diagnósticos, eu vou comparar os baselines com Poisson, binomial negativa e métodos temporais de contagem. Eu vou manter a mesma divisão primária 50/50 e a mesma validação temporal interna para isolar o efeito da família do modelo.

### Teste4 — robustez do SVR

Eu vou avaliar se a vantagem pontual do SVR se repete com diferentes sementes e janelas de origem. Para essa etapa, eu vou manter o baseline do último mês e o SARIMA como referências. Eu vou relatar a variação de MAE/RMSE e viés, além das métricas médias.

Eu não vou apresentar várias divisões sobrepostas como testes independentes. A avaliação fora da amostra precisa permanecer temporal e usar medidas padronizadas, como recomenda a revisão mais recente de [Mategula et al. (2025)](https://doi.org/10.1371/journal.pgph.0003751).

## 4. Priorizar métodos de contagem com dependência temporal

Os modelos atuais incluem GLM Poisson e binomial negativa com defasagens, mas uma regressão com defasagens pode deixar autocorrelação nos resíduos. Eu vou priorizar estes próximos candidatos:

1. **GLARMA Poisson e binomial negativa** ou **INGARCH/INAR**, que incorporam dependência temporal na própria estrutura da série de contagens.
2. **GSARIMA/GARMA binomial negativa**, para combinar sazonalidade anual, dependência temporal e distribuição de contagem.
3. **Modelo Bayesiano dinâmico binomial negativo**, se eu precisar comunicar intervalos preditivos e incerteza além da previsão média.

[Briët et al. (2013)](https://doi.org/10.1371/journal.pone.0065761) desenvolveram GSARIMA/GARIMA para contagens de malária e motivam os dois primeiros grupos. [Diao et al. (2023)](https://doi.org/10.3390/ijerph20136303) comparam Poisson e binomial negativa com histórico de casos; eu uso esse estudo para justificar as duas distribuições, sem transferir para Breves o resultado obtido com clima e dados do Senegal.

Eu vou revisar autocorrelação dos resíduos, dispersão condicional e estabilidade dos parâmetros antes de interpretar um modelo de contagem como adequado. A razão variância/média bruta atual é aproximadamente 147, mas a tendência e mudanças de regime podem inflar essa razão; ela é um sinal para investigar, não um diagnóstico definitivo.

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

1. Revisar definição, contagens e disponibilidade temporal dos dados.
2. Preservar `Teste1` como comparação principal atual.
3. Executar `Teste2` com 70/30 e registrar a estabilidade do ranking.
4. Implementar GLARMA/INGARCH ou GSARIMA binomial negativa e comparar em uma nova pasta.
5. Reavaliar SVR e SARIMA por janelas/seed sem consultar resultados futuros.
6. Integrar clima e covariáveis somente após confirmar cobertura e disponibilidade no momento da previsão.
7. Validar a conclusão final com meses novos ou outra série municipal compatível.

Eu registro o protocolo, os artigos e os resultados de cada etapa em [`revisao_e_protocolo.md`](revisao_e_protocolo.md) e em suas pastas `results/TesteN/` correspondentes.

## Referências que orientam este roteiro

- Briët OJT, Vounatsou P, Gunawardena DM, Galappaththy GNL, Amerasinghe PH. (2008). Models for short term malaria prediction in Sri Lanka. *Malaria Journal*, 7:76. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/). Eu uso para justificar validação temporal fora da amostra, inclusão de métodos clássicos e checagem de zeros/dados ausentes.
- Briët OJT, Amerasinghe PH, Vounatsou P. (2013). Generalized Seasonal Autoregressive Integrated Moving Average Models for Count Data with Application to Malaria Time Series with Low Case Numbers. *PLOS ONE*, 8(6):e65761. [DOI](https://doi.org/10.1371/journal.pone.0065761). Eu uso para priorizar GSARIMA/GARMA com distribuição de contagem.
- Zinszer K, Verma AD, Charland K, et al. (2012). A scoping review of malaria forecasting: past work and future directions. *BMJ Open*, 2:e001992. [DOI](https://doi.org/10.1136/bmjopen-2012-001992). Eu uso para manter os métodos na mesma série e reportar métricas comuns.
- Diao O, Absil P-A, Diallo M. (2023). Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal. *International Journal of Environmental Research and Public Health*, 20(13):6303. [DOI](https://doi.org/10.3390/ijerph20136303). Eu uso para comparar regressões Poisson e binomial negativa com histórico de casos.
- Sewe MO, Tozan Y, Ahlm C, Rocklöv J. (2017). Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya. *Scientific Reports*, 7:2589. [DOI](https://doi.org/10.1038/s41598-017-02560-z). Eu uso como motivação para testar clima e GAMBoost quando houver covariáveis alinhadas.
- Barboza D, et al. (2022). Prediction of malaria using deep learning models: A case study on city clusters in the state of Amazonas, Brazil, from 2003 to 2018. *Revista da Sociedade Brasileira de Medicina Tropical*. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9344950/). Eu uso como referência regional para uma etapa futura com várias séries municipais, não como justificativa para iniciar com LSTM/GRU em Breves.
- Mategula D, et al. (2025). Advancing Early Warning Systems for Malaria: Progress, challenges, and future directions — A scoping review. *PLOS Global Public Health*, 5(5):e0003751. [DOI](https://doi.org/10.1371/journal.pgph.0003751). Eu uso para reforçar avaliação fora da amostra e diagnóstico de autocorrelação.
