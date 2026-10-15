# Leitura dos gráficos históricos

Eu revisei `comparacao_metodos_1.png`, `comparacao_metodos_2.png` e `comparacao_metodos_3.png`, preservados em `results/legacy_optimizers/`. Eles mostram avaliação de fevereiro de 2013 a setembro de 2026, conforme os próprios títulos. Os gráficos indicam:

| Método | MAE (casos) | RMSE (casos) |
|---|---:|---:|
| Baseline: último mês | **25,8** | **35,6** |
| Random Forest | 28,9 | 39,9 |
| Média móvel de 3 meses | 29,4 | 41,6 |
| Ensemble otimizado por PSO | 32,7 | 42,4 |
| Ensemble otimizado por GA | 33,2 | 43,0 |
| Ensemble otimizado por BAT | 33,4 | 43,3 |
| SVR | 37,0 | 48,0 |
| Média de 3 redes neurais | 37,5 | 50,7 |
| Rede neural PCA | 37,9 | 52,1 |
| Baseline sazonal de 12 meses | 60,5 | 81,4 |

## Minha interpretação

Nos gráficos disponíveis, **o melhor método foi repetir o valor do último mês**. Ele teve o menor MAE e o menor RMSE dentre os métodos exibidos. Random Forest foi o modelo de aprendizado de máquina mais próximo, mas ainda teve erros maiores. GA, PSO e BAT não melhoraram o ensemble em relação ao baseline; PSO foi o melhor entre os três otimizadores, mas continuou atrás do último mês.

O desenho das curvas ajuda a explicar esse resultado: no período de avaliação, muitos picos são seguidos por uma queda rápida e o valor do mês anterior frequentemente acompanha o nível local. A previsão sazonal de 12 meses falha em várias mudanças de nível, o que elevou bastante seu erro. As redes e o SVR parecem mais instáveis ou suavizam/subestimam partes dos picos.

Essa conclusão é **descritiva e restrita aos PNGs antigos**. As figuras não documentam uma comparação 50/50 com ajuste interno idêntico para todos os modelos, e as saídas vieram de execuções e versões diferentes do código. Portanto, eu não trato o baseline como vencedor definitivo da pesquisa. A comparação nova em `results/Teste1/` fixa a mesma série, divisão temporal, horizonte mensal, métricas e validação interna para todos os candidatos.

## Resultado da comparação nova

Na execução atual, o **SVR foi o melhor por MAE (24,71) e RMSE (35,31)**. O baseline do último mês teve 27,07/37,18 e o SARIMA 27,59/35,39. A diferença de RMSE entre SVR e SARIMA foi de apenas 0,08 caso; a diferença pareada de MAE do SVR contra o baseline teve IC95% [-5,82; 0,99], incluindo zero. Por isso eu chamo o SVR de melhor resultado pontual, sem concluir que sua vantagem seja estatisticamente robusta. O SARIMA teve a menor deviance de Poisson (12,35) e o SVR subestimou em média 6,44 casos; a escolha pode depender do objetivo de alerta.

Os resultados completos e a justificativa dos artigos, incluindo como aplico cada um, estão em [docs/revisao_e_protocolo.md](docs/revisao_e_protocolo.md) e em [results/Teste1/](results/Teste1/).

## O que eu faria a seguir

1. Compararia primeiro baselines simples, Poisson/NB temporal, ETS, ARIMA/SARIMA e modelos de ML sob o mesmo teste cronológico.
2. Verificaria se Poisson é apropriado comparando média e variância e examinando os resíduos; se houver sobredispersão, daria prioridade à binomial negativa.
3. Testaria GLARMA/INGARCH e GSARIMA para modelar a dependência temporal de contagens que o GLM com defasagens pode deixar nos resíduos.
4. Só então adicionaria clima com defasagens, desde que as medições estejam alinhadas ao momento real da previsão.
5. Manteria GA/PSO/BAT como comparação secundária de busca de hiperparâmetros, com orçamento igual, sem apresentá-los como previsores.

Eu fundamentei a escolha de métodos e do protocolo em estudos de previsão de malária que comparam séries temporais, regressões de contagem e validação fora da amostra. Em especial, [Briët et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2412896/) usam uma divisão temporal de aproximadamente metade/metade e mostram que o melhor método varia por distrito; [Zinszer et al. (2012)](https://doi.org/10.1136/bmjopen-2012-001992) recomendam comparar métodos na mesma base com medidas comuns; e [Diao et al. (2023)](https://doi.org/10.3390/ijerph20136303) avaliam Poisson e binomial negativa para contagens de malária. A revisão e a explicação de como uso cada artigo estão em [docs/revisao_e_protocolo.md](docs/revisao_e_protocolo.md).
