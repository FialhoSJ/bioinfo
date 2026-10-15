# Revisão da literatura e protocolo da comparação

## Pergunta da pesquisa

Eu quero responder: **qual método prevê melhor o número mensal de casos notificados de malária em Breves (PA), usando apenas o histórico de notificações que já temos?** O alvo desta etapa é a soma mensal de `Número de registros` por data de notificação. O CSV ainda não inclui clima; por isso, não atribuo a ele o efeito de chuva, temperatura ou umidade. Os resultados aqui são previsões de contagens observadas, não incidência populacional nem risco causal.

Eu procurei estudos revisados por pares em periódicos de saúde pública, epidemiologia e doenças tropicais que avaliam previsões temporais de malária. Selecionei os trabalhos abaixo porque comparam métodos, avaliam previsões fora da amostra ou sintetizam boas práticas que se aplicam à pergunta local.

## Artigos encontrados e como os aplico

| Artigo e periódico | O que o estudo fez e encontrou | Como eu uso no projeto |
|---|---|---|
| Briët et al. (2008), *Malaria Journal*, [Models for short term malaria prediction in Sri Lanka](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/) | Comparou suavização exponencial, ARIMA/SARIMA e modelos com e sem chuva em contagens mensais de distritos. Ajustou em aproximadamente a primeira metade da série e avaliou a segunda metade; o melhor método variou por distrito e horizonte. | Uso como precedente para separar cronologicamente treino e teste e para incluir baselines, ETS e ARIMA/SARIMA. A série deles era mais longa e os parâmetros ficaram fixos no teste; aqui a comparação é de um passo mensal e reajusta o modelo conforme cada observação real ficaria disponível. O artigo também reforça que não há um vencedor universal. |
| Briët, Amerasinghe e Vounatsou (2013), *PLOS ONE*, [Generalized Seasonal ARIMA Models for Count Data](https://doi.org/10.1371/journal.pone.0065761) | Propôs GSARIMA/GARIMA bayesiano com distribuição binomial negativa para contagens sazonais e não estacionárias de malária. A avaliação fora da amostra favoreceu GSARIMA por MARE frente ao modelo sazonal determinístico considerado; os autores alertam que aproximações gaussianas podem falhar em contagens baixas. | Uso para justificar modelos que respeitem a natureza de contagem e para recomendar GSARIMA ou GLARMA como próximos candidatos. O código atual testa Poisson e binomial negativa com defasagens, mas ainda não implementa a estrutura completa GSARIMA/GARMA. |
| Zinszer et al. (2012), *BMJ Open*, [A scoping review of malaria forecasting](https://doi.org/10.1136/bmjopen-2012-001992) | Revisou 29 estudos. Encontrou grande variedade de modelos e métricas, o que impedia comparar diretamente a acurácia entre artigos; recomendou comparar métodos diferentes na mesma base e usar medidas comuns. | Uso como justificativa central para executar todos os candidatos nos mesmos meses de teste e publicar MAE, RMSE, MASE, viés e deviance de Poisson. Não comparo números de artigos diferentes como se fossem uma disputa direta. |
| Diao, Absil e Diallo (2023), *International Journal of Environmental Research and Public Health*, [Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal](https://doi.org/10.3390/ijerph20136303) | Comparou regressões Poisson e binomial negativa para contagens mensais, usando histórico de casos e covariáveis climáticas e programáticas. Poisson com ligação identidade foi mais adequado naquele conjunto; a escolha de ligação e defasagens importou, e retirar o histórico de casos piorou as previsões. | Uso para incluir Poisson e binomial negativa com defasagens de casos e para testar ligações log e identidade. Como ainda não temos clima nem intervenções alinhadas, não reproduzo os efeitos ou conclusões desses preditores. O resultado do Senegal não determina o vencedor em Breves. |
| Sewe et al. (2017), *Scientific Reports*, [Using remote sensing environmental data to forecast malaria incidence](https://doi.org/10.1038/s41598-017-02560-z) | Comparou GAM e GAMBOOST para admissões mensais de crianças em hospital do Quênia, com clima por satélite e horizontes de um a três meses. GAMBOOST teve menor erro no teste de um mês naquele local. | Uso como referência para adicionar GAM/GAMBoost e testar clima em uma etapa futura. O resultado não é transferível diretamente para Breves porque a população, a variável de resposta e os preditores diferem, e a implementação atual não contém aquelas covariáveis. |
| Barboza et al. (2022), *Revista da Sociedade Brasileira de Medicina Tropical*, [Prediction of malaria using deep learning models in Amazonas](https://pmc.ncbi.nlm.nih.gov/articles/PMC9344950/) | Comparou LSTM, GRU e Random Forest em agrupamentos de municípios do Amazonas, com série de 2003 a 2018; o desempenho diferiu entre agrupamentos e métodos. | Uso como referência regional para uma experiência posterior com modelos recorrentes. Não trato LSTM/GRU como primeira escolha agora: após a separação 50/50 restam 112 meses de desenvolvimento para um único município, pouco dado para redes profundas e muito menos informação espacial que no estudo. |
| Mategula et al. (2025), *PLOS Global Public Health*, [Advancing Early Warning Systems for Malaria](https://doi.org/10.1371/journal.pgph.0003751) | Revisou 30 estudos recentes; encontrou validação e métricas heterogêneas e alertou que regressões podem deixar autocorrelação nos resíduos sem modelagem. | Uso para incluir validação fora da amostra com métricas padronizadas e para colocar o diagnóstico de resíduos/autocorrelação como melhoria. Também sustenta testar modelos que incorporem explicitamente dependência temporal. |

## A divisão de 50% é adequada?

Eu escolhi 50% inicial para desenvolvimento e 50% final para teste cronológico como **primeira comparação**, não como regra universal. Há precedente direto em Briët et al., que ajustaram em cerca de metade da série e avaliaram a outra metade. A revisão de Zinszer et al. também mostra que a reserva de uma parte dos dados é prática comum, mas a conclusão científica depende de aplicar todos os modelos à mesma divisão e usar métricas comuns.

No CSV atual há 237 meses completos, de janeiro de 2007 a setembro de 2026. Com 12 meses de histórico inicial, formam-se 225 alvos de um mês à frente. Como esse total é ímpar, a divisão mais próxima de metade reserva 112 alvos para desenvolvimento, até abril de 2017, e 113 para teste, de maio de 2017 a setembro de 2026. Assim, a metade inicial contém mais de nove anos de alvos: razoável para comparar modelos parcimoniosos, mas ainda pequena para justificar redes profundas ou modelos com muitos parâmetros. Outubro de 2026 foi excluído porque a última notificação do CSV é de 5 de outubro, logo o mês está incompleto.

Na agregação atual, encontrei 30.249 linhas válidas de Breves-PA, 30.236 registros somados em 237 meses, nenhum mês sem linha que precisasse ser preenchido e nenhuma contagem mensal igual a zero. A média mensal é 127,6; a variância bruta é 18.768,8, cerca de 147 vezes a média. Essa razão bruta sugere forte dispersão e dá motivo para testar a binomial negativa, mas não prova por si só que o Poisson condicional seja inadequado: tendência, sazonalidade e mudanças ao longo de quase vinte anos também elevam a variância. Eu vou verificar isso com resíduos dos modelos ajustados.

No teste, eu avalio 113 previsões **mensais de um passo à frente**: para prever maio de 2017 uso somente dados anteriores a maio; depois que o valor real de maio estaria disponível, ele pode entrar no ajuste de junho. As configurações e hiperparâmetros ficam selecionados somente com validação temporal interna nos primeiros 112 alvos. Isso representa atualização operacional mensal. Não significa treinar uma vez e prever recursivamente todos os 113 meses sem observar os valores reais intermediários.

Essa decisão é eficaz para responder a uma pergunta operacional de curto prazo e dá um teste extenso em diferentes períodos. Ela tem limites: a primeira metade tem mudanças de regime e talvez não represente a segunda; além disso, ao revisar repetidamente os resultados desse teste e alterar os modelos, o teste deixa de ser intocado. Por isso, eu trato este resultado como comparação exploratória forte, não confirmação final. Como análise de sensibilidade, posso executar `Teste2` com 70/30, mas não escolherei a fração que produza o melhor ranking; o protocolo primário permanece registrado como 50/50. A confirmação final ideal usará meses futuros ainda não examinados.

## O que comparei nesta versão

Todos os métodos recebem a mesma série-alvo agregada e são comparados nos mesmos meses. A validação interna usa três dobras temporais de 12 meses somente dentro da metade de desenvolvimento; seleciono configurações pelo RMSE interno. No teste externo, mantenho a configuração escolhida e reajusto cada método com o passado disponível em cada origem mensal.

| Família | Métodos nesta execução | Por que entrou |
|---|---|---|
| Baselines | último mês; média móvel de 3 meses; mesmo mês do ano anterior | Mostram o nível de desempenho que modelos complexos precisam superar. O gráfico antigo indica que o último mês já é um competidor forte. |
| Suavização e séries temporais gaussianas | ETS/Holt-Winters; ARIMA; SARIMA | Comparam tendência e sazonalidade com métodos clássicos usados em previsões de malária. Os resultados de previsão são truncados em zero, mas estes modelos não são modelos de contagem. |
| Regressão de contagem | Poisson e binomial negativa com defasagens 1, 3 e 12 e ciclo mensal | Produzem médias não negativas e modelam a distribuição de contagens. Poisson é referência; binomial negativa permite sobredispersão. |
| Aprendizado supervisionado | Random Forest; HistGradientBoosting com perda Poisson; SVR | Verificam se relações não lineares em defasagens e calendário ajudam com o volume de dados disponível. Eu interpreto esses modelos com cautela por haver só 112 alvos para desenvolvimento. |

As entradas da etapa principal são contagens mensais defasadas (lag 1 = mês anterior; lag 12 = mesmo mês do ano anterior), médias móveis, tendência curta e seno/cosseno do mês do ano. Eu não uso sexo, faixa etária ou espécie como preditores nesta etapa: são partições contemporâneas da própria contagem e a data em que estariam disponíveis para uma previsão real ainda não foi estabelecida. Também não uso clima, população ou intervenções, ausentes ou ainda não alinhados no CSV atual.

## Como vou declarar o melhor método

Eu ordeno os métodos por RMSE no teste e reporto MAE junto, pois ele é interpretável como erro médio em casos. Também verifico MASE, viés, sMAPE e deviance de Poisson. Um método só merece ser chamado de melhor se a melhoria for consistente nas métricas principais e nas previsões mensais, e se superar os baselines de forma relevante. A diferença pareada de erro absoluto contra o baseline do último mês e seu intervalo bootstrap em blocos de 12 meses ajudam a mostrar incerteza; com um único município e dependência serial, não interpreto esses intervalos como prova definitiva.

As contagens podem receber previsões fracionárias porque o modelo prevê valor esperado. Os gráficos devem ser examinados para picos subestimados, mudanças de nível, viés e períodos em que todos os modelos falham. Correlação isolada não escolhe o modelo: ela mede co-movimento, não a magnitude do erro.

### Resultado da primeira comparação (`Teste1`)

Eu rodei o protocolo padrão com 112 meses de desenvolvimento (até abril de 2017) e 113 meses de teste (maio de 2017 a setembro de 2026). Pela métrica principal RMSE, o SVR ficou em primeiro por uma margem mínima sobre o SARIMA; pelo MAE, o SVR também ficou em primeiro. Assim, **o SVR foi o vencedor pontual desta execução**, com configuração `C=50` e `epsilon=1`.

| Método | MAE (casos) | RMSE (casos) | Viés (casos) | Deviance Poisson |
|---|---:|---:|---:|---:|
| SVR | **24,71** | **35,31** | -6,44 | 12,61 |
| SARIMA | 27,59 | 35,39 | +9,46 | **12,35** |
| ARIMA | 26,59 | 36,16 | +0,32 | 12,97 |
| ETS | 27,21 | 37,14 | +0,09 | 13,64 |
| Último mês | 27,07 | 37,18 | -0,17 | 13,80 |
| Binomial negativa com lags | 28,68 | 38,44 | -1,39 | 15,59 |
| Boosting Poisson | 27,90 | 38,90 | +2,15 | 13,65 |
| Random Forest | 28,26 | 39,09 | +2,53 | 14,19 |
| Média móvel de 3 meses | 30,78 | 42,15 | +0,28 | 17,57 |
| Poisson com lags | 35,90 | 43,99 | +14,16 | 21,28 |
| Mesmo mês do ano anterior | 54,73 | 70,22 | +9,69 | 49,42 |

O SVR reduziu o MAE em 2,36 casos (8,7%) e o RMSE em 1,87 casos (5,0%) frente ao baseline do último mês. A diferença pareada de MAE foi -2,36 casos, mas o intervalo bootstrap em blocos de 12 meses foi [-5,82; 0,99] e inclui zero. Portanto, eu não afirmo que o SVR seja comprovadamente superior ao baseline. Além disso, seu viés foi negativo: subestimou em média 6,44 casos. O SARIMA teve RMSE quase idêntico, mas viés positivo maior; também obteve a melhor deviance de Poisson. A escolha depende da métrica e do custo de subestimar surtos.

Uma explicação plausível — ainda não demonstrada por análise de importância ou ablação — é que o SVR com núcleo RBF aproveitou relações não lineares entre contagens defasadas, médias móveis e mês do ano no horizonte de um mês. A figura de previsões mostra que ele acompanha o nível geral, mas amortece alguns picos. O teste não contém clima, portanto não permite concluir nada sobre contribuição climática.

O Poisson com lags teve erros altos e viés positivo, enquanto a binomial negativa melhorou bastante sobre o Poisson, mas não venceu os baselines ou os métodos temporais. A razão variância/média bruta de 147 reforça a necessidade de modelos de contagem com sobredispersão, embora tendência e mudanças de regime também influenciem essa razão. Esses resultados motivam GLARMA/INGARCH e GSARIMA, em linha com [Briët et al. (2013)](https://doi.org/10.1371/journal.pone.0065761), que propõem estrutura sazonal explicitamente para contagens. Também mantenho o SARIMA como candidato forte porque [Briët et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/) encontraram que o melhor método mudava conforme distrito e horizonte.

O ranking do `Teste1` continua exploratório: usei um único corte histórico, e a diferença do SVR frente ao baseline não foi clara no intervalo pareado. Ao repetir escolhas após consultar o teste, eu reduzo sua independência; uma confirmação mais forte exige dados futuros. Os arquivos completos, previsões e figuras estão em [`results/Teste1/`](../results/Teste1/). A necessidade de avaliar métodos na mesma série com métricas comuns segue a recomendação de [Zinszer et al. (2012)](https://doi.org/10.1136/bmjopen-2012-001992).

Para revisar os valores e as curvas, eu deixei a [tabela de métricas](../results/Teste1/metricas_por_metodo.csv), as [previsões do teste](../results/Teste1/previsoes_teste.csv), o [gráfico de RMSE](../results/Teste1/rmse_por_metodo.png) e o [gráfico de previsões](../results/Teste1/previsoes_teste.png) junto com as configurações e a validação interna.

## O que ainda pode melhorar e quais métodos priorizar

### Prioridade 1 — qualidade e disponibilidade dos dados

1. Eu confirmaria com a fonte se cada mês sem linha significa zero casos ou falha de extração/vigilância. O código preenche lacunas com zero e lista os meses inseridos; zero só é válido quando essa interpretação é confirmada.
2. Eu documentaria atraso e revisão das notificações. Uma previsão feita no início do mês não pode usar casos que ainda não tinham sido registrados naquele momento.
3. Eu verificaria se a soma de `Número de registros` não duplica linhas agregadas e preservaria a definição da população e do tipo de caso. Sem denominador, não chamarei a saída de incidência.
4. Eu adicionaria chuva, temperatura e umidade em uma fase posterior, com defasagens biologicamente plausíveis e disponíveis na data de previsão. Testaria cada fonte em uma ablação para medir se ela realmente acrescenta informação ao histórico de casos.
5. Eu avaliaria mudanças no sistema de notificação, intervenções e população. A série cobre quase vinte anos e os padrões de transmissão podem mudar; uma quebra estrutural pode tornar o passado distante menos útil.

### Prioridade 2 — candidatos estatísticos de contagem

1. **GLARMA Poisson/binomial negativa** ou **INGARCH/INAR**: são meus próximos candidatos de maior prioridade porque modelam a dependência serial dentro de uma distribuição de contagem. O Poisson/NB GLM implementado usa casos defasados como covariáveis, mas não necessariamente remove toda a autocorrelação dos resíduos.
2. **GSARIMA/GARMA binomial negativa**, conforme Briët et al. (2013): representa sazonalidade e dependência temporal mantendo uma distribuição de contagem; precisa ser avaliado se a implementação e o tempo de ajuste forem viáveis.
3. **Modelo Bayesiano dinâmico binomial negativo com tendência e sazonalidade**: permite propagar incerteza e acomodar mudanças graduais. Eu o priorizaria se a finalidade incluir intervalos preditivos para decisão de saúde pública.
4. **GAM com resposta Poisson/NB**, e depois GAMBoost: permite sazonalidade e efeitos suaves; GAMBoost entra após confirmar estabilidade em validação temporal. O estudo de Sewe et al. motiva a experiência, mas seu melhor desempenho dependeu de dados ambientais que ainda não temos.

### Prioridade 3 — aprendizado de máquina

Eu manteria Random Forest e boosting Poisson como referências não lineares e acrescentaria CatBoost/LightGBM com objetivo de contagem apenas se a amostra crescer ou se agregarmos municípios de modo coerente. LSTM/GRU fica para uma etapa com mais séries ou painel municipal, janelas de treino suficientes e comparação com baselines fortes. O estudo brasileiro de Barboza et al. usou agrupamentos de municípios; não é evidência de que uma rede profunda seja melhor para a única série de Breves.

## Otimizadores: devo usá-los?

Para a pergunta central — qual método prevê melhor os casos? — eu não colocaria GA, PSO e Bat Algorithm como métodos preditivos concorrentes. Eles não estimam a série por si mesmos: buscam atributos ou hiperparâmetros para outro preditor. Nos gráficos antigos, os ensembles selecionados por GA, PSO e BAT tiveram MAE/RMSE maiores que o baseline do último mês, e a seleção não mostrou ganho prático naquela execução.

Eu usaria ajuste por máxima verossimilhança, penalização e validação temporal para estimar e escolher configurações dos modelos estatísticos. Para modelos de ML, usaria uma grade pequena predefinida ou busca aleatória/otimização bayesiana **dentro da validação interna**, com orçamento comparável. Só manteria GA/PSO/BAT como uma pergunta secundária — “qual estratégia de busca ajusta melhor o mesmo preditor sob o mesmo orçamento?” — com sementes repetidas, orçamento de avaliações igual e teste externo intocado. Isso separa desempenho do preditor do efeito do otimizador.

## Organização e artefatos

- `src/malaria_forecasting/`: leitura, agregação, modelos, avaliação e interface de linha de comando.
- `data/raw/`: CSV fornecido; ignorado pelo Git por ser grande.
- `results/Teste1/`: primeira comparação padrão 50/50.
- `results/Teste2/`: espaço para comparação de sensibilidade, por exemplo 70/30.
- `results/legacy_optimizers/` e `results/legacy_hybrid/`: arquivos antigos preservados, sem misturá-los aos resultados novos.
- `run.sh`: executa a CLI e escolhe um nome diferente para cada pasta.

Cada execução nova grava métricas, previsões mensais, configurações escolhidas, validação interna, diferenças pareadas, resumo JSON e gráficos. Eu mantenho um diretório por execução para que não haja sobrescrita silenciosa.

## Referências

- Briët OJT, Vounatsou P, Gunawardena DM, Galappaththy GNL, Amerasinghe PH. (2008). Models for short term malaria prediction in Sri Lanka. *Malaria Journal*, 7:76. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/).
- Briët OJT, Amerasinghe PH, Vounatsou P. (2013). Generalized Seasonal Autoregressive Integrated Moving Average Models for Count Data with Application to Malaria Time Series with Low Case Numbers. *PLOS ONE*, 8(6):e65761. [DOI](https://doi.org/10.1371/journal.pone.0065761).
- Zinszer K, Verma AD, Charland K, et al. (2012). A scoping review of malaria forecasting: past work and future directions. *BMJ Open*, 2:e001992. [DOI](https://doi.org/10.1136/bmjopen-2012-001992).
- Diao O, Absil P-A, Diallo M. (2023). Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal. *International Journal of Environmental Research and Public Health*, 20(13):6303. [DOI](https://doi.org/10.3390/ijerph20136303).
- Sewe MO, Tozan Y, Ahlm C, Rocklöv J. (2017). Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya. *Scientific Reports*, 7:2589. [DOI](https://doi.org/10.1038/s41598-017-02560-z).
- Barboza D, et al. (2022). Prediction of malaria using deep learning models: A case study on city clusters in the state of Amazonas, Brazil, from 2003 to 2018. *Revista da Sociedade Brasileira de Medicina Tropical*. [Texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9344950/).
- Mategula D, et al. (2025). Advancing Early Warning Systems for Malaria: Progress, challenges, and future directions — A scoping review. *PLOS Global Public Health*, 5(5):e0003751. [DOI](https://doi.org/10.1371/journal.pgph.0003751).
