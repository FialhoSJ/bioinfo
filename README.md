# Previsão mensal de casos de malária em Breves (PA)

Neste projeto eu comparo métodos para prever a contagem mensal de registros autóctones de malária em Breves, Pará. A série usada nesta etapa contém notificações e não possui as covariáveis climáticas; por isso, os modelos recebem somente contagens passadas e mês do ano.

Organizei o código em `src/malaria_forecasting/`, os dados de entrada em `data/raw/`, a revisão e o protocolo em `docs/`, e cada execução em uma pasta própria de `results/`. Os PNGs e CSVs anteriores foram mantidos em `results/legacy_optimizers/` e `results/legacy_hybrid/`; eles não são resultados da avaliação nova.

## Preparar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Coloque o CSV indicado em [data/raw/README.md](data/raw/README.md). O arquivo bruto é ignorado pelo Git por seu tamanho.

## Executar

```bash
./run.sh Teste1
```

Cada execução cria `results/<nome>/` sem sobrescrever resultados existentes. Exemplos:

```bash
./run.sh Teste2 --train-fraction 0.7
./run.sh TesteRapido --quick
./run.sh EstudoAlternativo --data /caminho/arquivo.csv --seed 123
```

O padrão divide os 225 alvos mensais elegíveis em 112 para desenvolvimento e 113 para teste, em ordem temporal. A primeira metade seleciona as configurações por validação temporal interna; a segunda metade é avaliada com previsões mensais sucessivas de um mês à frente. Depois que o valor real de um mês estaria disponível, ele entra no histórico para prever o mês seguinte. Esse protocolo não é uma previsão recursiva de nove anos feita de uma só vez.

Os métodos incluem baselines temporais, ETS, ARIMA/SARIMA, regressões Poisson e binomial negativa com defasagens de casos, Random Forest, boosting Poisson e SVR. O código exporta métricas, previsões, configurações, validação interna, comparação pareada e gráficos em cada pasta de teste.

## O que encontrei até aqui

Nos gráficos antigos `comparacao_metodos_*.png`, o baseline do último mês obteve MAE 25,8 e RMSE 35,6, menores que os valores mostrados para RF (28,9/39,9), média móvel de três meses (29,4/41,6), ensembles com GA/PSO/BAT, SVR e redes neurais. Portanto, naquele experimento, o baseline venceu os modelos mais complexos. Como esses artefatos vieram de outra configuração e não têm a divisão cronológica de 50/50 desta etapa, trato essa leitura como diagnóstico histórico, não como resultado final.

Na nova execução `Teste1`, o SVR obteve o menor MAE (24,71) e RMSE (35,31), mas a diferença pareada contra o baseline do último mês não foi conclusiva. O SARIMA teve RMSE quase igual (35,39) e a menor deviance de Poisson. A análise detalhada está na revisão metodológica.

O protocolo, a revisão de artigos, as limitações dos dados e o plano de melhoria estão em [docs/revisao_e_protocolo.md](docs/revisao_e_protocolo.md). A interpretação detalhada dos gráficos antigos está em [analise_resultados_modelos.md](analise_resultados_modelos.md).
