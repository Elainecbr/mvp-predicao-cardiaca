# Registro dos testes e da refatoração da API

## Contexto

Em 27 de novembro de 2026, foi criado um backup do projeto antes das alterações. Os ambientes virtuais, o histórico `.git` e caches foram excluídos do ZIP; o código, os arquivos de dados e modelo, o notebook e as alterações locais foram incluídos. A integridade do arquivo compactado foi verificada com `unzip -tq`.

O objetivo foi separar a rota Flask, a validação e a predição em módulos, **sem retreinar o modelo nem mudar a entrada e saída da API**.

## Testes automatizados

Com o ambiente virtual descrito no [README](../README.md), na raiz do projeto:

```bash
python -m pytest api/test_app.py -q
```

Resultado da execução local: **67 testes passaram**, tanto antes quanto depois da separação dos módulos.

Os testes abrangem os mapeamentos explícitos de todas as categorias do notebook, a ordem das variáveis, limites numéricos aceitos e rejeitados, entradas inválidas, campos ausentes ou extras, CORS das origens locais, respostas da API e duas predições esperadas do modelo atual. Também verificam se acurácia e recall na partição de teste salva são pelo menos 0,80.

Esses dois limiares são verificações técnicas escolhidas para o protótipo, não critérios clínicos. As duas predições esperadas documentam o comportamento do **modelo atual**; se ele for intencionalmente substituído, seus resultados devem ser reavaliados, e os testes atualizados apenas após uma análise fundamentada.

## Testes de integração realizados

- **API sem Docker:** servidor Flask local temporário, em porta alternativa, respondeu `{"prediction": 1}` a um exemplo de teste.
- **API com Docker:** contêiner reconstruído com os novos módulos respondeu `{"prediction": 1}` ao mesmo exemplo.
- **Formulário com Docker:** no navegador, o botão de exemplo preencheu 11 campos e o envio à API exibiu uma classificação experimental.
- **Artefatos de treinamento:** os hashes SHA-256 do notebook `Predicao_Doenca_Cardiaca_A.ipynb`, do pipeline e dos dois arquivos de teste permaneceram iguais antes e depois da alteração.

Esses testes verificam os casos exercitados, não provam ausência de todos os defeitos, generalização para novas populações ou validade para uso médico. O notebook não foi reexecutado no Colab nesta etapa. Os resultados da avaliação dependem da partição salva, que não substitui uma avaliação externa independente.

## Organização atual

- [app.py](../api/app.py): configuração Flask e rota `/predict`.
- [validacao.py](../api/validacao.py): validação dos campos, ordem e mapeamentos categóricos.
- [predicao.py](../api/predicao.py): carregamento do pipeline e predição.
- [test_app.py](../api/test_app.py): testes da API, dos mapeamentos e do modelo.

Os mapeamentos continuam fora do pipeline e precisam permanecer consistentes com o notebook. A possível experiência com um notebook B e pré-processamento dentro do pipeline ficará para outra etapa.
