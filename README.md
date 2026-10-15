# Bioinfo — previsão mensal de malária

Pesquisa de métodos de previsão mensal de registros autóctones de malária em Breves, Pará. A implementação compara algoritmos genéricos de busca (GA, PSO e Bat Algorithm), modelos preditivos (Random Forest, SVR e redes neurais) e baselines temporais. O método híbrido experimental foi retirado da avaliação atual.

## Preparar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No VS Code/Jupyter, selecione o kernel **Python (.venv)**.

## Dados

O código procura `6- Autóctones Mun Fx Etária e Sexo_Dados completos_data.csv` na pasta do projeto e em `~/Downloads`. Também é possível indicar outro caminho com `--data`. O arquivo de dados não é incluído neste repositório.

A variável-alvo é a contagem mensal de registros notificados, sem denominador populacional. O reindexamento de meses sem registros como zero pressupõe ausência real de casos; essa hipótese precisa ser confirmada na fonte de vigilância. A disponibilidade, na origem da previsão, das contagens demográficas do mês de origem e os atrasos de notificação também precisam ser avaliados.

## Protocolo principal

O protocolo cobre alvos de 2008 ao último ano configurado (2026 por padrão). Os primeiros 60 meses elegíveis formam o treinamento inicial; assim, as métricas fora da amostra começam depois desses 60 meses. A avaliação externa avança em blocos de 12 meses e expande o treinamento a cada bloco. Dentro de cada bloco, a previsão é feita mês a mês, reajustando os modelos com os dados que já estariam observados.

GA, PSO e BAT selecionam atributos e hiperparâmetros para o mesmo ensemble RF/SVR/PCA-ANN. Os três recebem a mesma população inicial e o mesmo número de indivíduos e iterações. O ensemble é comparado com Random Forest, SVR, uma rede PCA-MLP, a média de três redes e baselines de último mês, média móvel de três meses e mesmo mês do ano anterior. Os modelos isolados usam configurações de referência fixas e não devem ser interpretados como comparações de modelos igualmente ajustados.

A busca usa somente os dados anteriores ao bloco avaliado, com três dobras temporais internas de 12 meses. Cada bloco externo é repetido com três sementes. São exportados os erros por bloco e semente, previsões mensais, convergência e diferenças pareadas contra o baseline do último mês. O ranking médio é descritivo; só o menor RMSE pontual não estabelece superioridade. Consulte o protocolo e as limitações em [analise_resultados_modelos.md](analise_resultados_modelos.md).

## Executar a análise

Com o ambiente ativado e o CSV disponível no projeto ou em `~/Downloads`:

```bash
python3 hybrid_forecasting.py \
  --start-year 2008 \
  --end-year 2026 \
  --initial-train-months 60 \
  --outer-horizon 12 \
  --repetitions 3 \
  --cv-splits 3 \
  --cv-test-months 12 \
  --population-size 4 \
  --iterations 3
```

O nome `hybrid_forecasting.py` foi mantido por compatibilidade com o projeto; o método híbrido anterior não está mais implementado na comparação. Use `--data /caminho/arquivo.csv` para especificar a fonte ou `--output /caminho/avaliacao.png` para escolher o painel principal. Reduzir as sementes ou o orçamento de busca diminui o custo, mas a configuração reduzida deve ser identificada como análise exploratória.

O módulo grava ao lado do painel:

- `avaliacao_diagnostica.png`: previsões, resíduos, convergência e comparação pareada;
- `comparacao_metodos_*.png`: séries de previsões separadas por método;
- `comparacao_otimizadores.png`: barras para GA, PSO e BAT;
- `resultados_modelos.csv` e `metricas_agregadas.csv`: médias e dispersões das métricas;
- `metricas_por_janela_semente.csv`: resultados de cada bloco externo e semente;
- `previsoes_mensais.csv` e `previsoes_por_janela_semente.csv`;
- `comparacoes_pareadas.csv`: diferença de erro absoluto vs. último mês e intervalo bootstrap em blocos de 12 meses;
- `convergencia_otimizadores.csv` e `selecao_atributos_parametros.csv` para convergência e auditoria da busca.

O notebook [bioinfo.ipynb](bioinfo.ipynb) contém as fases exploratórias e o mesmo protocolo principal. A execução do notebook também grava esses arquivos no diretório de trabalho.

## Interpretação

MAE e RMSE são expressos em casos; valores menores são melhores. MASE compara o erro ao erro sazonal calculado somente no treinamento. Viés positivo indica superestimação. sMAPE/MAPE e correlação são métricas complementares. Uma diferença pareada negativa favorece o método sobre o último mês; intervalo que inclui zero não dá suporte claro a uma diferença. Como há várias comparações e dependência temporal, os intervalos são exploratórios e devem ser discutidos com cautela.
