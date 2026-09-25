# Sistema de Recomendação de Jogadores de Futebol

MVP desenvolvido como parte de um Trabalho de Conclusão de Curso em Ciência da Computação.

O projeto utiliza dados estatísticos de desempenho e informações de valor de mercado para identificar possíveis substitutos para um jogador de referência.

O objetivo é encontrar jogadores tecnicamente compatíveis e, posteriormente, considerar o aspecto financeiro para priorizar alternativas potencialmente mais viáveis para clubes com menor poder de investimento.

## Status

**MVP funcional.**

O sistema atualmente possui:

* Modelos específicos para atacantes, meio-campistas, defensores e goleiros;
* Filtro mínimo de minutos jogados;
* Padronização das métricas utilizadas em cada posição;
* Cálculo de proximidade técnica entre jogadores;
* Seleção de candidatos tecnicamente compatíveis;
* Integração de valores de mercado;
* Ranking técnico-financeiro;
* API desenvolvida com FastAPI;
* Endpoint para busca de jogadores;
* Endpoint para geração de recomendações.

## Tecnologias

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* KaggleHub

## Estrutura do projeto

```text
TCC Futebol/
│
├── data/
│   └── players_data-2024_2025_com_valores.csv
│
├── src/
│   ├── api.py
│   ├── recomendacao.py
│   └── preparar_valores_mercado.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

O arquivo:

```text
data/players_data-2024_2025_com_valores.csv
```

corresponde à base final utilizada pelo sistema de recomendação.

Esse arquivo já contém os dados esportivos necessários e os valores de mercado associados aos jogadores, permitindo executar diretamente o MVP sem precisar repetir toda a etapa de preparação dos dados.

Os demais arquivos CSV utilizados durante coleta, tratamento e associação dos dados são ignorados pelo Git.

## Funcionamento do sistema

O sistema utiliza métricas diferentes de acordo com a posição do jogador.

Antes da comparação, as variáveis são padronizadas utilizando o `StandardScaler`.

A proximidade entre o jogador utilizado como referência e os demais jogadores é calculada utilizando distância euclidiana sobre as características padronizadas.

O processo de recomendação ocorre, de forma simplificada, nas seguintes etapas:

1. Seleção dos jogadores da mesma categoria posicional;
2. Aplicação do filtro mínimo de 900 minutos jogados;
3. Padronização das métricas estatísticas;
4. Cálculo da distância técnica entre os jogadores;
5. Seleção dos 30 candidatos tecnicamente mais próximos;
6. Normalização do desempenho técnico dentro desse grupo;
7. Aplicação de um corte técnico relativo mínimo de 50%;
8. Remoção de candidatos sem valor de mercado disponível;
9. Cálculo do componente financeiro;
10. Combinação dos componentes técnico e financeiro;
11. Ordenação dos jogadores pelo índice final de recomendação.

O ranking final utiliza:

```text
60% componente técnico
40% componente financeiro
```

O valor de mercado não é utilizado como característica técnica do jogador.

Primeiro são identificados jogadores tecnicamente compatíveis. Somente depois o aspecto financeiro influencia a ordenação final dos candidatos.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/bmlogs64/TCC-Futebol.git
```

Entre na pasta do projeto:

```bash
cd "TCC Futebol"
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` contém:

```text
kagglehub
pandas
numpy
scikit-learn
fastapi
uvicorn
```

## Executando a API

Entre na pasta `src`:

```bash
cd src
```

Execute:

```bash
uvicorn api:app --reload
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

A documentação interativa do FastAPI pode ser acessada em:

```text
http://127.0.0.1:8000/docs
```

Por meio dessa página é possível testar os endpoints diretamente pelo navegador.

## Buscar jogadores

Endpoint:

```http
GET /jogadores
```

Esse endpoint permite pesquisar jogadores utilizando parte do nome.

Exemplo:

```text
nome = mbapp
```

Resposta:

```json
{
  "busca": "mbapp",
  "total_encontrados": 1,
  "jogadores": [
    {
      "nome": "Kylian Mbappé",
      "clube": "Real Madrid",
      "posicao": "FW",
      "idade": 25,
      "valor_mercado": "€ 180.000.000"
    }
  ]
}
```

Esse endpoint pode ser utilizado para localizar o nome exato do jogador antes de solicitar uma recomendação.

## Gerar recomendações

Endpoint:

```http
GET /recomendacoes
```

Parâmetros:

```text
jogador
modelo
quantidade
clube
```

Os modelos disponíveis são:

```text
atacante
meio
defensor
goleiro
```

O parâmetro `clube` é opcional e pode ser utilizado caso existam jogadores com nomes iguais.

Exemplo:

```text
jogador = Kylian Mbappé
modelo = atacante
quantidade = 10
```

Exemplo de resposta:

```json
{
  "jogador_referencia": {
    "nome": "Kylian Mbappé",
    "clube": "Real Madrid",
    "valor_mercado": "€ 180.000.000"
  },
  "criterios": {
    "peso_tecnico": "60%",
    "peso_financeiro": "40%",
    "corte_tecnico_relativo": "50%"
  },
  "total_recomendacoes": 6,
  "recomendacoes": [
    {
      "posicao_ranking": 1,
      "jogador": "Leroy Sané",
      "clube": "Bayern Munich",
      "idade": 28,
      "valor_mercado": "€ 32.000.000",
      "vantagem_financeira": "€ 148.000.000",
      "similaridade_tecnica": "51,11%",
      "indice_recomendacao": "93,96%"
    }
  ]
}
```

A quantidade de jogadores retornados pode ser menor que a quantidade solicitada.

Isso ocorre porque o sistema não reduz os critérios técnicos apenas para completar uma quantidade fixa de recomendações.

## Interpretação dos resultados

### Similaridade técnica

O campo:

```text
similaridade_tecnica
```

representa um índice calculado a partir da distância técnica entre o candidato e o jogador de referência.

Quanto maior o valor, maior a proximidade entre os perfis técnicos analisados.

### Vantagem financeira

O campo:

```text
vantagem_financeira
```

representa a diferença entre o valor de mercado do jogador de referência e o valor do candidato.

Exemplo:

```text
€ 20.000.000
```

indica que o candidato custa aproximadamente € 20 milhões a menos.

Um valor negativo:

```text
-€ 5.000.000
```

indica que o candidato custa aproximadamente € 5 milhões a mais que o jogador utilizado como referência.

### Índice de recomendação

O campo:

```text
indice_recomendacao
```

é utilizado para definir a posição final do jogador no ranking.

Ele combina o componente técnico e o componente financeiro segundo os pesos definidos no MVP.

## Dados

O repositório mantém versionado somente o arquivo final necessário para execução do MVP:

```text
data/players_data-2024_2025_com_valores.csv
```

Os arquivos intermediários utilizados durante coleta e preparação dos dados não são enviados ao repositório.

Dessa forma, o dataset final necessário para executar a aplicação permanece disponível no GitHub, enquanto arquivos temporários ou intermediários continuam ignorados.

## Objetivo do MVP

O MVP busca demonstrar como técnicas de Ciência de Dados podem auxiliar o processo de scouting no futebol.

A proposta não consiste em simplesmente recomendar os jogadores mais baratos.

O sistema primeiro identifica atletas tecnicamente compatíveis com o jogador utilizado como referência e, posteriormente, utiliza informações financeiras para priorizar alternativas potencialmente mais viáveis.

Dessa forma, o projeto busca oferecer apoio baseado em dados para situações em que um clube precisa encontrar possíveis substitutos mantendo um equilíbrio entre desempenho esportivo e capacidade financeira.