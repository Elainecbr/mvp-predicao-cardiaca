# Roteiro de apresentação — até 3 minutos

Este roteiro é uma proposta para ensaiar. A regra da disciplina é apresentar um vídeo de **no máximo três minutos**, mostrando a aplicação em execução e chamando o modelo. Use um cronômetro e ajuste o ritmo; os tempos abaixo são aproximados.

## Antes de gravar

- Deixe o notebook já aberto no Colab e seus resultados visíveis. Não é necessário executar todo o treinamento durante a gravação.
- Inicie a API e o front-end seguindo o [README](../README.md), com ou sem Docker. Confira previamente que o formulário consegue obter uma resposta.
- Deixe abertas as abas do notebook, da aplicação e, se desejar, do terminal ou repositório. Use somente o botão de **exemplo fictício** para a demonstração.
- Confira as métricas exibidas na versão final do notebook. Os números abaixo descrevem os artefatos presentes no projeto no momento da elaboração deste roteiro; se houver novo treinamento, atualize-os.
- Ajuste o zoom para o texto ficar legível e grave um ensaio para confirmar que a narração e a demonstração cabem no tempo.

## 0:00–0:25 — Apresentação e problema

**Mostrar:** título do projeto no notebook ou no formulário.

> Olá! Sou Elaine e este é meu MVP acadêmico de classificação de doença cardíaca, desenvolvido no módulo Engenharia de Sistemas de Software Inteligentes da PUC-Rio. O objetivo é estudar o ciclo de machine learning e sua integração com uma aplicação web. É um protótipo de experimentação, não uma ferramenta de diagnóstico ou triagem real.

## 0:25–1:05 — Dados e treinamento

**Mostrar:** carga do dataset, divisão dos dados e comparação de modelos no Colab.

> Utilizei o conjunto público Heart Failure Prediction, com 918 registros e 11 atributos de entrada. O notebook carrega o arquivo por uma URL pública e separa os dados em treino e teste de forma estratificada. Comparei KNN, árvore de classificação, Naive Bayes, SVM e Random Forest, com dados sem escala, padronizados e normalizados. Usei validação cruzada e busca de hiperparâmetros para escolher um modelo.

## 1:05–1:35 — Resultado e limites

**Mostrar:** avaliação e matriz de confusão.

> Neste experimento, o KNN padronizado foi selecionado. Na partição de teste, obteve aproximadamente 88% de acurácia e 91% de recall para a classe positiva. Dos 102 casos positivos, nove não foram identificados. Esses resultados não garantem desempenho em outras populações. Há também valores zero em algumas variáveis que exigem investigação. O modelo não foi validado clinicamente.

## 1:35–2:25 — Aplicação funcionando

**Mostrar:** formulário, botão de exemplo e resultado; se houver tempo, mostrar a rota `/predict` no código.

> A aplicação tem front-end e API Flask em diretórios separados. Vou preencher o formulário com um exemplo fictício e pedir a classificação. O JavaScript envia os 11 campos para a API; ela valida e transforma as entradas e chama o pipeline treinado, que retorna a classe exibida na tela. Não são necessários dados pessoais para esta demonstração.

**Ação:** clique em **Preencher exemplo fictício** e depois em **Avaliar com o modelo**. Espere e mostre a resposta. Se ocorrer erro, não apresente uma imagem estática como se fosse uma predição ao vivo.

## 2:25–2:50 — Testes e encerramento

**Mostrar:** teste do modelo ou terminal com o resultado de uma execução real.

> O PyTest verifica a acurácia e o recall do modelo com a partição de teste salva no projeto. A aplicação pode ser executada localmente, com ou sem Docker. O código, o notebook e as instruções de execução estarão no repositório público. Obrigada!

## Conferência final

- Vídeo medido: **até 3:00**, com margem para transições.
- Aplicação e resultado de uma chamada real ao modelo aparecem na gravação.
- Não apresentar o resultado como diagnóstico, risco futuro comprovado ou validação clínica.
- Antes da entrega, publicar todos os arquivos necessários e testar o link público em uma janela anônima.
