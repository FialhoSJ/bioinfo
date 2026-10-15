# Avaliação metodológica dos modelos de previsão mensal

## Pergunta e escopo

O estudo prevê a quantidade de registros de malária autóctone notificados em Breves-PA no mês seguinte, a partir de informações disponíveis até o mês de origem. A resposta é uma **contagem**, não uma classe de risco. Por isso, acurácia, sensibilidade, especificidade e MCC não são métricas para a tarefa atual.

O período configurado começa em 2008 e termina em 2026. Os primeiros 60 meses elegíveis são reservados ao treinamento inicial; a avaliação fora da amostra começa após esse período e pode se estender até o último mês completo disponível. Portanto, “2008–2026” descreve a janela dos dados, não significa que 2008 seja também o primeiro mês pontuado. Se a série utilizável começar em janeiro de 2008, a primeira previsão externa ocorre após os 60 meses de treinamento.

## Protocolo de avaliação

1. A janela externa começa com 60 meses de treinamento e avança em blocos de 12 meses, aumentando o histórico de treinamento a cada bloco.
2. Dentro de cada janela externa, a seleção de atributos e parâmetros dos ensembles usa apenas a parte anterior ao bloco avaliado. São usadas três dobras temporais internas, cada uma com 12 meses de validação. Com o treinamento mínimo de 60 meses, a primeira dobra interna ainda dispõe de 24 meses de ajuste.
3. No bloco externo, cada previsão é de um mês à frente. Após o mês real ficar disponível, o modelo é reajustado para prever o mês seguinte. Essa política representa uma operação de previsão sequencial.
4. A busca é repetida com três sementes em cada janela externa. GA, PSO e Bat Algorithm recebem a mesma população inicial e o mesmo tamanho de população e número de iterações. A implementação registra chamadas e avaliações únicas da função objetivo.
5. Previsões, métricas por janela/semente, curvas de convergência e diferenças pareadas são exportadas, para que a comparação possa ser auditada.

Essa separação entre busca interna e avaliação externa reduz o viés de avaliar um modelo nos mesmos dados que orientaram a escolha de seus parâmetros. Ela não elimina incerteza, dependência serial, mudança de regime ou o risco de escolhas repetidas após observar os resultados.

## Métodos comparados

### Otimizadores genéricos

GA, PSO e Bat Algorithm são **algoritmos de busca**, não preditores independentes. Cada um escolhe uma máscara de atributos e parâmetros para o mesmo ensemble com Random Forest, SVR e PCA-ANN. A função objetivo é o RMSE médio das dobras internas acrescido de uma penalização proporcional à fração de atributos selecionados. Assim, a comparação entre GA, PSO e BAT tenta isolar o efeito da estratégia de busca.

O orçamento é alinhado pelo tamanho inicial da população e pelas iterações; os três métodos começam dos mesmos candidatos. O cache evita refazer avaliações idênticas, e o número de candidatos únicos é registrado. Comparações futuras devem também informar o total de ajustes de modelos e o tempo efetivo, pois uma avaliação pode custar diferentemente conforme o tamanho do subconjunto escolhido.

### Modelos de referência

O ensemble otimizado é comparado com Random Forest, SVR, uma rede PCA-MLP e a média de três redes PCA-MLP com sementes diferentes. Esses modelos isolados usam todos os atributos e configurações prespecificadas. São referências úteis para diagnóstico, mas **não estão ajustados por busca separada**; por isso, a comparação deles com o ensemble otimizado não é tão equilibrada quanto a comparação GA–PSO–BAT. O próximo refinamento deve ajustar cada modelo de referência dentro das mesmas dobras internas, sem consultar os blocos externos.

### Baselines temporais

- último mês observado;
- média dos três meses anteriores;
- mesmo mês do ano anterior.

Os baselines são essenciais: um método complexo só é útil se melhorar previsões simples sob o mesmo protocolo temporal.

## Métricas e leitura

- **MAE:** erro absoluto médio em casos, fácil de interpretar.
- **RMSE:** erro em casos que penaliza mais os erros grandes; métrica usada para o ranking pontual.
- **MASE:** MAE dividido pelo erro sazonal calculado apenas no treinamento. Valor abaixo de 1 indica que o MAE foi menor que a escala sazonal de referência.
- **Viés:** média de previsão menos real. Positivo significa superestimação.
- **sMAPE e MAPE:** métricas percentuais complementares. MAPE ignora meses com valor real zero e precisa ser lido com essa limitação.
- **Correlação:** descrição de co-movimento; não mede o tamanho dos erros e não deve ser usada sozinha para escolher um modelo.

O resumo mostra a média e o desvio-padrão entre blocos externos e sementes. A diferença pareada compara, mês a mês, o erro absoluto de cada método com o erro absoluto do baseline do último mês. Diferença negativa favorece o método. O intervalo é calculado por bootstrap circular em blocos de 12 meses, com 2.000 reamostragens. Um intervalo que inclui zero não sustenta uma conclusão clara de superioridade. Como vários métodos são comparados e há poucos blocos temporais independentes, os intervalos são exploratórios; devem ser acompanhados de limitações e não usados como prova isolada.

## Avaliação dos resultados disponíveis

Os números atualmente presentes nos PNGs e CSVs antigos são **históricos e exploratórios**; ainda não foram recalculados sob o protocolo descrito acima. Há resultados de janelas e versões diferentes: uma tabela anterior de 60 meses aponta o baseline do último mês com MAE 22,30 e RMSE 31,07, enquanto um gráfico posterior de outra execução aponta PSO com MAE 21,8 e RMSE 30,9. O gráfico também mostra GA com MAE 24,4/RMSE 34,4 e BAT com MAE 26,6/RMSE 33,1. A diferença pequena entre baseline e PSO, a ausência de repetições e a incompatibilidade entre artefatos impedem concluir que PSO seja melhor.

Além disso, as tabelas antigas do teste de Friedman usavam apenas alguns modelos e folds internos; não comparavam corretamente os três otimizadores. Esse teste foi removido do fluxo principal. A nova exportação permite uma análise pareada no período externo, que corresponde à pergunta de desempenho futuro.

**Nenhuma nova métrica confirmatória é relatada neste documento.** Os CSVs antigos não devem ser apresentados como saída da avaliação nova. Após executar o módulo ou a Fase 5 do notebook, revisar `metricas_agregadas.csv`, `metricas_por_janela_semente.csv` e `comparacoes_pareadas.csv`, e então preencher a interpretação com os valores efetivamente produzidos.

## Limitações e melhorias prioritárias

1. **Validar meses sem linhas.** O código preenche meses ausentes com zero. Isso só é correto se a fonte confirmar que houve vigilância completa e nenhum caso; ausência de notificação ou falha de extração não equivale a zero.
2. **Definir disponibilidade operacional dos atributos.** Contagens por espécie, sexo e idade no mês de origem podem chegar depois do mês em questão. Sem carimbo de quando a informação ficou disponível, esse preditor pode representar uma previsão retrospectiva, não uma previsão utilizável em tempo real.
3. **Ajustar modelos isolados.** RF/SVR/ANN de referência ainda usam hiperparâmetros fixos. Ajustá-los dentro da validação interna torna mais justa sua comparação com os ensembles selecionados por GA/PSO/BAT.
4. **Avaliar atraso e revisão das notificações.** O alvo mede registros por data de notificação; atraso, subnotificação e revisões podem alterar a série histórica.
5. **Inspecionar as MLPs.** Avisos de não convergência e instabilidade entre sementes devem ser relatados; pode ser necessário simplificar a rede ou comparar com regressão regularizada.
6. **Adicionar fontes externas apenas com alinhamento temporal.** Chuva, temperatura, população e mudanças no sistema de notificação podem melhorar a interpretação, mas precisam estar disponíveis no momento da origem e ter cobertura compatível.
7. **Separar seleção e confirmação.** Após observar repetidamente os resultados externos e mudar o método, o período deixa de ser um holdout intocado. A confirmação final ideal exige dados futuros ou um bloco final previamente reservado.
8. **Controlar a multiplicidade.** Se forem feitas inferências formais sobre muitos modelos ou métricas, definir contrastes primários antes da análise e considerar ajuste para múltiplas comparações.

## Arquivos produzidos pelo protocolo

- `avaliacao_diagnostica.png`: previsão, resíduos, convergência e diferenças pareadas;
- `comparacao_metodos_*.png`: painéis temporais por método;
- `comparacao_otimizadores.png`: gráfico de barras GA, PSO e BAT;
- `resultados_modelos.csv` e `metricas_agregadas.csv`: resumo de métricas;
- `metricas_por_janela_semente.csv`: desempenho por bloco e repetição;
- `previsoes_mensais.csv` e `previsoes_por_janela_semente.csv`;
- `comparacoes_pareadas.csv`: diferenças mensais pareadas e intervalo bootstrap;
- `convergencia_otimizadores.csv`: histórico da função objetivo;
- `selecao_atributos_parametros.csv`: atributos e parâmetros selecionados por janela e semente.

## Referências

- Hyndman, R. J.; Athanasopoulos, G. *Forecasting: Principles and Practice*, seção sobre validação cruzada temporal. [Texto online](https://otexts.com/fpptr/tscv.html).
- Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: An analysis and review*. [International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207000000650).
- Cerqueira, V.; Torgo, L.; Mozetič, I. (2022). *Evaluating time series forecasting models: An empirical study on performance estimation methods*. [Data Mining and Knowledge Discovery](https://link.springer.com/article/10.1007/s10618-022-00894-5).
- P2CA-PSO-ANN para previsão de casos de malária com meteorologia. [Artigo](https://doi.org/10.1016/j.iswa.2022.200164).
- RF-iSMA-SVM para classificação de gravidade clínica. [Artigo](https://doi.org/10.1016/j.compbiomed.2025.110302).
- TRSMA-SVM para seleção de atributos em classificação. [Artigo](https://doi.org/10.1016/j.eij.2025.100709).
- TMFS-MTS para avaliação de seleção de atributos em séries multivariadas. [Artigo](https://doi.org/10.1016/j.simpa.2025.100789).
