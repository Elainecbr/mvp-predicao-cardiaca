# Guia de estudo — notebook, modelo e aplicação

Este guia acompanha a versão do [notebook](../api/MachineLearning/notebooks/Predicao_Doenca_Cardiaca_A.ipynb) e a aplicação existentes no projeto. É material de apoio: o roteiro curto para a entrega está em [roteiro-video.md](roteiro-video.md).

## 1. O objetivo do MVP

O projeto é um **protótipo acadêmico** que estuda a classificação de registros associados a doença cardíaca e a integração de um modelo com uma aplicação web. Ele não é uma ferramenta validada para diagnóstico, decisões de saúde ou triagem real; a classe retornada também não é uma estimativa comprovada de risco futuro.

O conjunto utilizado é o [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), distribuído no projeto como [heart.csv](../heart.csv). A cópia tem 918 linhas, 11 colunas de entrada e a coluna-alvo `HeartDisease`.

## 2. Notebook no Colab: passo a passo

### Contexto e bibliotecas

As células de texto apresentam o problema, o conjunto de dados e a metodologia. Em seguida, as bibliotecas são importadas: `pandas` para dados, `matplotlib` para gráficos, Scikit-Learn para modelos e métricas, e `pickle` para exportar o modelo.

**Como explicar:** “O notebook é também o relatório do experimento: ele mostra não só o código, mas a justificativa de cada etapa.”

### Carga e análise exploratória

O CSV é lido por uma URL pública do GitHub. As células `info`, `describe` e o gráfico da classe alvo ajudam a conhecer os tipos das colunas, estatísticas e proporção das classes.

**Atenção:** existem 172 valores zero em `Cholesterol` e um em `RestingBP`. Esses valores merecem investigação, mas não devemos afirmar sua origem sem confirmação na documentação da fonte. Em uma evolução, pode-se comparar uma estratégia de tratamento de valores ausentes dentro do pipeline, mantendo o CSV original.

### Codificação das categorias

As categorias textuais são convertidas em números com mapeamentos explícitos: por exemplo, `Sex` mapeia `M` para 1 e `F` para 0. Isso não cria dados clínicos novos: representa categorias existentes em formato numérico. Porém, a codificação ordinal pode impor distâncias artificiais entre categorias para o KNN; vale comparar outras codificações em trabalho futuro.

**Ponto essencial:** esses mapeamentos são realizados no notebook **antes** do pipeline e reproduzidos na API. Portanto, o arquivo `.pkl` contém a padronização e o classificador, mas **não contém esses mapeamentos de texto**. Alterar um mapeamento ou a ordem das 11 colunas exige manter notebook, API e testes consistentes.

### Holdout e validação cruzada

`X` contém as entradas; `y`, a resposta conhecida. O notebook reserva 20% para teste e usa 80% para treino, com `stratify` para preservar aproximadamente as proporções das classes. A semente fixa ajuda na reprodução da divisão. No treino, a validação cruzada estratificada divide os dados em dez partes para comparar os candidatos.

**Como explicar:** “O conjunto de teste fica reservado até a avaliação final; a escolha dos parâmetros usa o conjunto de treino.”

### Comparação dos modelos e escalas

O notebook compara KNN, árvore de classificação (CART), Naive Bayes, SVM e Random Forest. Cada um é avaliado sem escala, com `StandardScaler` e com `MinMaxScaler`: são 15 combinações.

`Pipeline` reúne as etapas: durante a validação cruzada, o escalonador é ajustado usando apenas a parte de treino de cada divisão. Isso evita um tipo importante de vazamento de dados. O gráfico boxplot ajuda a comparar os resultados entre as partições.

### Busca de hiperparâmetros

`GridSearchCV` compara combinações para SVM, Random Forest e KNN usando os dados de treino e validação cruzada. O modelo escolhido é o melhor entre **esses três candidatos otimizados**, segundo acurácia média de validação cruzada; não significa que todas as combinações possíveis de todos os algoritmos foram examinadas.

### Avaliação final

O modelo selecionado faz predições sobre a partição reservada. A **acurácia** é a proporção de classificações corretas; o **recall da classe positiva** é a proporção de positivos identificados; o **F1** combina precisão e recall. A matriz de confusão ajuda a ver acertos e erros por classe.

Nos artefatos atuais: KNN com padronização, sete vizinhos e distância Manhattan; acurácia de aproximadamente 88,04% e recall de 91,18% no teste. A matriz é `[[69, 13], [9, 93]]`: dos 102 positivos dessa amostra, 93 foram identificados e 9 não. Valores próximos entre validação e teste são encorajadores, mas não provam ausência de overfitting nem generalização para outras populações.

### Exportação

O notebook exporta o pipeline e dois arquivos: `X_test_cardiaco.csv`, com entradas codificadas, e `y_test_cardiaco.csv`, com as respostas. O teste automatizado compara as predições do modelo salvo com essas respostas. A partição de teste já foi usada na avaliação do experimento: testes repetidos nela **não substituem** uma avaliação independente.

## 3. Por que o exemplo dos professores tem mais arquivos?

No exemplo de diabetes, há modelos individuais, um `MinMaxScaler` separado e um pipeline. A rota de predição carrega especificamente `rf_diabetes_pipeline.pkl`; o teste de modelos também carrega os modelos de regressão logística e KNN individualmente. Eles representam diferentes formas e experimentos de armazenamento, não uma exigência de usar todos os arquivos em cada predição.

No nosso projeto, `modelo_cardiaco_pipeline.pkl` contém **um `StandardScaler` já ajustado e o KNN já treinado**. Quando chamamos `predict`, a própria esteira aplica a padronização e, em seguida, a classificação. Portanto, não precisamos de outro arquivo `minmax_scaler` para esse modelo. A normalização `MinMaxScaler` foi experimentada no notebook, mas não é a etapa do pipeline escolhido. Salvar outros candidatos pode ser útil para comparação, mas não é necessário para executar o modelo vencedor.

`X_test` e `y_test` são arquivos separados dos professores e também do nosso projeto. Não são o modelo: são entradas e respostas para avaliá-lo. A API recebe os dados novos pelo formulário; não usa esses arquivos de teste para responder às requisições.

**Responsabilidades:** o exemplo separa classes de pré-processamento, carregamento e avaliação. No nosso projeto, [app.py](../api/app.py) cuida da rota, [validacao.py](../api/validacao.py) valida e codifica os campos e [predicao.py](../api/predicao.py) carrega e executa o pipeline; [test_app.py](../api/test_app.py) verifica esses comportamentos. Dividir arquivos por si só não melhora o modelo nem exige vários `.pkl`. Uma evolução futura seria experimentar o pré-processamento categórico dentro do pipeline treinado; isso exige novo treinamento e validação.

## 4. Aplicação full stack

1. **Front-end:** [index.html](../front/index.html) apresenta os campos; [styles.css](../front/styles.css) define o visual; [scripts.js](../front/scripts.js) monta o JSON e faz a chamada HTTP. O botão de exemplo preenche valores fictícios e não altera o treinamento.
2. **API Flask:** [app.py](../api/app.py) usa [validacao.py](../api/validacao.py) para validar e converter as 11 entradas e [predicao.py](../api/predicao.py) para carregar o pipeline confiável e executar `predict`. A rota responde com a classe 0 ou 1.
3. **Testes:** [test_app.py](../api/test_app.py) testa validação, integração e desempenho. Os limites atuais de acurácia e recall são 0,80, escolhidos para verificação técnica do protótipo, não como critérios de segurança clínica. O PyTest precisa ser executado: ele não bloqueia uma implantação automaticamente sem integração apropriada.
4. **Execução:** o [README](../README.md) apresenta instruções sem Docker e com Docker Compose, que inicia front-end e API em contêineres separados.

## 5. Limitações e cuidados ao apresentar

- A base é limitada e não demonstra desempenho em outras populações, regiões ou condições. Não inventar variáveis que a base não possui.
- Não afirmar que os resultados garantem ausência de overfitting, conformidade com a LGPD ou prontidão para produção.
- O formulário não solicita nome e não persiste as entradas, mas dados de saúde são sensíveis e a minimização, por si só, não garante anonimização ou conformidade legal.
- Arquivos `pickle` só devem ser carregados de fonte confiável. A versão do Scikit-Learn deve ser compatível com a usada ao salvar o modelo.
- O notebook ainda contém expressões como “apoio à triagem” e algumas conclusões categóricas. Revise-as antes da entrega para refletir o caráter experimental aqui descrito.
- Após publicar o código da aplicação, confira a execução independente no Colab, o funcionamento do link público do repositório e a duração do vídeo.
