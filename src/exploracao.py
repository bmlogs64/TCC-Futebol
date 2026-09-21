import kagglehub
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

import numpy as np

path = kagglehub.dataset_download(
    "hubertsidorowicz/football-players-stats-2024-2025"
)

print("Local do dataset:")
print(path)

print("\nArquivos encontrados:")

for arquivo in os.listdir(path):
    print("-", arquivo)

arquivo_csv = os.path.join(
    path,
    "players_data-2024_2025.csv"
)

df = pd.read_csv(arquivo_csv)

print("\n===== BASE CARREGADA =====")
print("Quantidade de registros:", len(df))
print("Quantidade de colunas:", len(df.columns))

print("\n===== PRIMEIROS JOGADORES =====")

print(
    df[
        [
            "Player",
            "Pos",
            "Squad",
            "Comp",
            "Age",
            "Min"
        ]
    ].head()
)

print("\n===== POSIÇÕES =====")

quantidade_posicoes = df["Pos"].value_counts()

print(quantidade_posicoes)

print("\n===== COMPETIÇÕES =====")

quantidade_competicoes = df["Comp"].value_counts()

print(quantidade_competicoes)

print("\n===== MINUTOS DISPUTADOS =====")

print(df["Min"].describe())

print("\n===== COMPARAÇÃO DE LIMITES DE MINUTOS =====")

limites = [450, 900, 1350]

for limite in limites:
    quantidade = len(df[df["Min"] >= limite])
    percentual = (quantidade / len(df)) * 100

    print(
        f"{limite} minutos: "
        f"{quantidade} registros "
        f"({percentual:.2f}%)"
    )

MINIMO_MINUTOS = 900

df_validos = df[
    df["Min"] >= MINIMO_MINUTOS
].copy()

print("\n===== BASE APÓS FILTRO DE MINUTOS =====")
print("Registros originais:", len(df))
print("Registros válidos:", len(df_validos))

print("\n===== JOGADORES ÚNICOS =====")

quantidade_registros = len(df_validos)
quantidade_jogadores = df_validos["Player"].nunique()

print("Quantidade de registros:", quantidade_registros)
print("Quantidade de jogadores únicos:", quantidade_jogadores)
print(
    "Diferença:",
    quantidade_registros - quantidade_jogadores
)

print("\n===== JOGADORES COM MAIS DE UM REGISTRO =====")

quantidade_por_jogador = df_validos["Player"].value_counts()

jogadores_repetidos = quantidade_por_jogador[
    quantidade_por_jogador > 1
]

print(jogadores_repetidos)

print(
    "\nQuantidade de jogadores com mais de um registro:",
    len(jogadores_repetidos)
)

print("\n===== DETALHES DOS JOGADORES REPETIDOS =====")

registros_repetidos = df_validos[
    df_validos["Player"].isin(jogadores_repetidos.index)
]

colunas_repetidos = [
    "Player",
    "Age",
    "Pos",
    "Squad",
    "Comp",
    "MP",
    "Starts",
    "Min"
]

registros_repetidos = registros_repetidos[
    colunas_repetidos
].sort_values(by="Player")

print(
    registros_repetidos.to_string(index=False)
)

print("\n===== MEIO-CAMPISTAS =====")

df_meias = df_validos[
    df_validos["Pos"].str.contains("MF", na=False)
].copy()

print("Quantidade de meio-campistas:", len(df_meias))

print("\nDistribuição das posições:")
print(df_meias["Pos"].value_counts())

print("\n===== JOGADOR DE REFERÊNCIA =====")

nome_referencia = "Kevin De Bruyne"

jogador_referencia = df_meias[
    df_meias["Player"] == nome_referencia
]

print(
    jogador_referencia[
        [
            "Player",
            "Nation",
            "Age",
            "Pos",
            "Squad",
            "Comp",
            "MP",
            "Starts",
            "Min",
            "90s"
        ]
    ].to_string(index=False)
)

metricas_candidatas = [
    "xG",
    "xAG",
    "Sh/90",
    "SoT/90",
    "Cmp%",
    "KP",
    "PPA",
    "PrgP",
    "SCA90",
    "GCA90",
    "Touches",
    "Succ%",
    "PrgC",
    "CPA",
    "Recov"
]

print("\n===== MÉTRICAS DO JOGADOR DE REFERÊNCIA =====")

print(
    jogador_referencia[
        ["Player"] + metricas_candidatas
    ].to_string(index=False)
)

print("\n===== CRIAÇÃO DAS MÉTRICAS POR 90 MINUTOS =====")

metricas_totais = [
    "xG",
    "xAG",
    "KP",
    "PPA",
    "PrgP",
    "Touches",
    "PrgC",
    "CPA",
    "Recov"
]

for metrica in metricas_totais:
    nova_coluna = metrica + "_90"

    df_meias[nova_coluna] = (
        df_meias[metrica] / df_meias["90s"]
    )

print(
    df_meias[
        [
            "Player",
            "90s",
            "xG",
            "xG_90",
            "xAG",
            "xAG_90",
            "KP",
            "KP_90",
            "PrgP",
            "PrgP_90"
        ]
    ][
        df_meias["Player"] == nome_referencia
    ].to_string(index=False)
)

features = [
    "xG_90",
    "xAG_90",
    "Sh/90",
    "SoT/90",
    "Cmp%",
    "KP_90",
    "PPA_90",
    "PrgP_90",
    "SCA90",
    "GCA90",
    "Touches_90",
    "Succ%",
    "PrgC_90",
    "CPA_90",
    "Recov_90"
]

print("\n===== VALORES AUSENTES NAS FEATURES =====")

valores_ausentes = df_meias[features].isna().sum()

print(valores_ausentes)

print("\n===== ESTATÍSTICAS DAS FEATURES =====")

estatisticas_features = df_meias[features].describe()

print(estatisticas_features.to_string())

scaler = StandardScaler()

scaler.fit(df_meias[features])

features_padronizadas = scaler.transform(
    df_meias[features]
)

print("\n===== MATRIZ PADRONIZADA =====")
print("Formato:", features_padronizadas.shape)

indice_kevin = df_meias.index[
    df_meias["Player"] == nome_referencia
][0]

print("\n===== ÍNDICE DO JOGADOR DE REFERÊNCIA =====")
print("Índice original:", indice_kevin)

posicao_kevin = df_meias.index.get_loc(indice_kevin)

print("\n===== POSIÇÃO NA MATRIZ PADRONIZADA =====")
print("Posição do Kevin:", posicao_kevin)

vetor_kevin = features_padronizadas[posicao_kevin]

print("\n===== VETOR PADRONIZADO DO KEVIN =====")
print(vetor_kevin)

print("\n===== FEATURES PADRONIZADAS DO KEVIN =====")

for feature, valor in zip(features, vetor_kevin):
    print(f"{feature}: {valor:.4f}")


posicao_teste = 0

jogador_teste = df_meias.iloc[posicao_teste]
vetor_teste = features_padronizadas[posicao_teste]

print("\n===== JOGADOR DE TESTE =====")
print("Jogador:", jogador_teste["Player"])
print("Clube:", jogador_teste["Squad"])
print("Posição:", jogador_teste["Pos"])

distancia = np.linalg.norm(
    vetor_kevin - vetor_teste
)

print("\n===== DISTÂNCIA EUCLIDIANA =====")
print(
    f"Distância entre {nome_referencia} e "
    f"{jogador_teste['Player']}: {distancia:.4f}"
)

distancias = np.linalg.norm(
    features_padronizadas - vetor_kevin,
    axis=1
)

print("\n===== DISTÂNCIAS CALCULADAS =====")
print("Quantidade de distâncias:", len(distancias))

df_meias["Distancia"] = distancias

print("\n===== DISTÂNCIA DO JOGADOR DE REFERÊNCIA =====")

print(
    df_meias[
        df_meias["Player"] == nome_referencia
    ][
        ["Player", "Squad", "Distancia"]
    ].to_string(index=False)
)

ranking = df_meias[
    df_meias["Player"] != nome_referencia
].copy()

ranking = ranking.sort_values(
    by="Distancia",
    ascending=True
)

print("\n===== TOP 10 JOGADORES MAIS SEMELHANTES =====")

print(
    ranking[
        [
            "Player",
            "Age",
            "Pos",
            "Squad",
            "Comp",
            "Min",
            "Distancia"
        ]
    ].head(10).to_string(index=False)
)

primeiro_colocado = ranking.iloc[0]

indice_primeiro = primeiro_colocado.name
posicao_primeiro = df_meias.index.get_loc(indice_primeiro)

vetor_primeiro = features_padronizadas[posicao_primeiro]

print("\n===== PRIMEIRO COLOCADO DO RANKING =====")
print("Jogador:", primeiro_colocado["Player"])
print("Clube:", primeiro_colocado["Squad"])
print("Distância:", primeiro_colocado["Distancia"])
print("Posição na matriz:", posicao_primeiro)

print("\n===== COMPARAÇÃO KEVIN x LO CELSO =====")

for feature, valor_kevin, valor_primeiro in zip(
    features,
    vetor_kevin,
    vetor_primeiro
):
    diferenca = abs(valor_kevin - valor_primeiro)

    print(
        f"{feature}: "
        f"Kevin={valor_kevin:.4f} | "
        f"Lo Celso={valor_primeiro:.4f} | "
        f"Diferença={diferenca:.4f}"
    )

print("\n===== CONTRIBUIÇÃO PARA A DISTÂNCIA =====")

for feature, valor_kevin, valor_primeiro in zip(
    features,
    vetor_kevin,
    vetor_primeiro
):
    diferenca = valor_kevin - valor_primeiro
    contribuicao = diferenca ** 2

    print(
        f"{feature}: "
        f"{contribuicao:.4f}"
    )

diferencas = vetor_kevin - vetor_primeiro

soma_quadrados = np.sum(
    diferencas ** 2
)

distancia_manual = np.sqrt(
    soma_quadrados
)

print("\n===== VERIFICAÇÃO DA DISTÂNCIA =====")
print(f"Soma dos quadrados: {soma_quadrados:.4f}")
print(f"Raiz quadrada: {distancia_manual:.4f}")
print(f"Distância calculada anteriormente: {primeiro_colocado['Distancia']:.4f}")

print("\n===== VERIFICAÇÃO DA FEATURE Succ% =====")

print(
    df_meias.loc[
        df_meias["Player"].isin(
            [nome_referencia, primeiro_colocado["Player"]]
        ),
        ["Player", "Squad", "Succ", "Att_stats_possession", "Succ%"]
    ].to_string(index=False)
)

print("\n===== VALIDAÇÃO DO Succ% =====")

for nome in [nome_referencia, primeiro_colocado["Player"]]:

    jogador = df_meias[
        df_meias["Player"] == nome
    ].iloc[0]

    percentual_calculado = (
        jogador["Succ"] /
        jogador["Att_stats_possession"]
    ) * 100

    print(
        f"{nome}: "
        f"calculado={percentual_calculado:.2f}% | "
        f"dataset={jogador['Succ%']:.2f}%"
    )

correlacoes = df_meias[features].corr()

print("\n===== MATRIZ DE CORRELAÇÃO =====")
print(
    correlacoes.round(2).to_string()
)

print("\n===== CORRELAÇÕES ALTAS =====")

limite_correlacao = 0.80

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        correlacao = correlacoes.iloc[i, j]

        if abs(correlacao) >= limite_correlacao:
            print(
                f"{features[i]} x {features[j]}: "
                f"{correlacao:.2f}"
            )

print("\n===== KP_90 x SCA90 =====")

print(
    df_meias[
        [
            "Player",
            "Squad",
            "KP_90",
            "SCA90"
        ]
    ]
    .sort_values(
        by="KP_90",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

percentil_kp = (
    (df_meias["KP_90"] <= jogador_referencia["KP"].iloc[0] /
     jogador_referencia["90s"].iloc[0]).mean()
    * 100
)

percentil_sca = (
    (df_meias["SCA90"] <= jogador_referencia["SCA90"].iloc[0]).mean()
    * 100
)

print("\n===== POSIÇÃO DO KEVIN NAS MÉTRICAS =====")
print(f"KP_90: percentil {percentil_kp:.2f}")
print(f"SCA90: percentil {percentil_sca:.2f}")

features_revisadas = [
    "xG_90",
    "xAG_90",
    "Sh/90",
    "SoT/90",
    "Cmp%",
    "KP_90",
    "PPA_90",
    "PrgP_90",
    "GCA90",
    "Touches_90",
    "Succ%",
    "PrgC_90",
    "CPA_90",
    "Recov_90"
]

scaler_revisado = StandardScaler()

scaler_revisado.fit(
    df_meias[features_revisadas]
)

features_revisadas_padronizadas = scaler_revisado.transform(
    df_meias[features_revisadas]
)

print("\n===== MODELO REVISADO =====")
print(
    "Formato:",
    features_revisadas_padronizadas.shape
)

vetor_kevin_revisado = features_revisadas_padronizadas[
    posicao_kevin
]

distancias_revisadas = np.linalg.norm(
    features_revisadas_padronizadas - vetor_kevin_revisado,
    axis=1
)

print("\n===== DISTÂNCIAS DO MODELO REVISADO =====")
print("Quantidade de distâncias:", len(distancias_revisadas))
print(
    "Distância do Kevin para ele mesmo:",
    distancias_revisadas[posicao_kevin]
)

ranking_revisado = df_meias.copy()

ranking_revisado["Distancia_Revisada"] = distancias_revisadas

ranking_revisado = ranking_revisado[
    ranking_revisado["Player"] != nome_referencia
].copy()

ranking_revisado = ranking_revisado.sort_values(
    by="Distancia_Revisada",
    ascending=True
)

print("\n===== TOP 10 - MODELO REVISADO =====")

print(
    ranking_revisado[
        [
            "Player",
            "Age",
            "Pos",
            "Squad",
            "Comp",
            "Distancia_Revisada"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

top_baseline = ranking.head(10).reset_index(drop=True)
top_revisado = ranking_revisado.head(10).reset_index(drop=True)

print("\n===== COMPARAÇÃO DOS RANKINGS =====")

for i in range(10):
    jogador_baseline = top_baseline.iloc[i]["Player"]
    jogador_revisado = top_revisado.iloc[i]["Player"]

    print(
        f"{i + 1}º | "
        f"Baseline: {jogador_baseline} | "
        f"Revisado: {jogador_revisado}"
    )

print("\n===== Sh/90 x SoT/90 =====")

jogadores_top = [
    nome_referencia
] + ranking.head(5)["Player"].tolist()

print(
    df_meias[
        df_meias["Player"].isin(jogadores_top)
    ][
        [
            "Player",
            "Squad",
            "Sh/90",
            "SoT/90",
            "xG_90"
        ]
    ]
    .sort_values(
        by="Sh/90",
        ascending=False
    )
    .to_string(index=False)
)

print("\n===== PrgC_90 x CPA_90 =====")

print(
    df_meias[
        df_meias["Player"].isin(jogadores_top)
    ][
        [
            "Player",
            "Squad",
            "PrgC_90",
            "CPA_90"
        ]
    ]
    .sort_values(
        by="PrgC_90",
        ascending=False
    )
    .to_string(index=False)
)

print("\n===== PrgP_90 x Touches_90 =====")

print(
    df_meias[
        df_meias["Player"].isin(jogadores_top)
    ][
        [
            "Player",
            "Squad",
            "PrgP_90",
            "Touches_90"
        ]
    ]
    .sort_values(
        by="Touches_90",
        ascending=False
    )
    .to_string(index=False)
)

nome_referencia = "Martin Ødegaard"

jogadores_encontrados = df_meias[
    df_meias["Player"] == nome_referencia
]

print("\n===== JOGADOR DE REFERÊNCIA =====")

print(
    jogadores_encontrados[
        [
            "Player",
            "Squad",
            "Comp",
            "Pos",
            "Min"
        ]
    ].to_string(index=False)
)

print(
    "Quantidade de registros encontrados:",
    len(jogadores_encontrados)
)

if len(jogadores_encontrados) == 1:

    indice_referencia = jogadores_encontrados.index[0]

    posicao_referencia = df_meias.index.get_loc(
        indice_referencia
    )

    vetor_referencia = features_revisadas_padronizadas[
        posicao_referencia
    ]

    print("\n===== REFERÊNCIA LOCALIZADA =====")
    print("Índice no DataFrame:", indice_referencia)
    print("Posição na matriz:", posicao_referencia)
    print("Quantidade de features:", len(vetor_referencia))

if len(jogadores_encontrados) == 1:

    distancias_referencia = np.linalg.norm(
        features_revisadas_padronizadas - vetor_referencia,
        axis=1
    )

    ranking_dinamico = df_meias.copy()

    ranking_dinamico["Distancia"] = distancias_referencia

    ranking_dinamico = ranking_dinamico[
        ranking_dinamico["Player"] != nome_referencia
    ].copy()

    ranking_dinamico = ranking_dinamico.sort_values(
        by="Distancia",
        ascending=True
    )

    print("\n===== TOP 10 DINÂMICO =====")

    print(
        ranking_dinamico[
            [
                "Player",
                "Age",
                "Pos",
                "Squad",
                "Comp",
                "Min",
                "Distancia"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
def recomendar_jogadores(
    nome_jogador,
    quantidade=10,
    clube=None
):
    jogadores_encontrados = df_meias[
        df_meias["Player"] == nome_jogador
    ]

    if len(jogadores_encontrados) == 0:
        print(
            f"Jogador '{nome_jogador}' não encontrado."
        )
        return None

    if len(jogadores_encontrados) > 1:

        if clube is None:
            print(
                f"Existem {len(jogadores_encontrados)} registros "
                f"para '{nome_jogador}'. Escolha um clube:"
            )

            print(
                jogadores_encontrados[
                    [
                        "Squad",
                        "Comp",
                        "Pos",
                        "Min"
                    ]
                ].to_string(index=False)
            )

            return None

        jogadores_encontrados = jogadores_encontrados[
            jogadores_encontrados["Squad"] == clube
        ]

        if len(jogadores_encontrados) == 0:
            print(
                f"Não foi encontrado '{nome_jogador}' "
                f"no clube '{clube}'."
            )
            return None

    indice = jogadores_encontrados.index[0]

    posicao = df_meias.index.get_loc(
        indice
    )

    vetor = features_revisadas_padronizadas[
        posicao
    ]

    distancias = np.linalg.norm(
        features_revisadas_padronizadas - vetor,
        axis=1
    )

    resultado = df_meias.copy()

    resultado["Distancia"] = distancias

    resultado = resultado[
        resultado["Player"] != nome_jogador
    ].copy()

    resultado = resultado.sort_values(
        by="Distancia",
        ascending=True
    )

    return resultado.head(quantidade)

teste_recomendacao = recomendar_jogadores(
    "Amine Gouiri",
    10,
    clube="Barcelona"
)

if teste_recomendacao is not None:

    print("\n===== TESTE DA FUNÇÃO =====")

    print(
        teste_recomendacao[
            [
                "Player",
                "Age",
                "Pos",
                "Squad",
                "Comp",
                "Min",
                "Distancia"
            ]
        ].to_string(index=False)
    )

df_atacantes = df[
    (df["Min"] >= 900) &
    (df["Pos"].str.contains("FW", na=False))
].copy()

print("\n===== ATACANTES =====")
print("Quantidade de registros:", len(df_atacantes))

print("\nPosições encontradas:")
print(
    df_atacantes["Pos"].value_counts()
)

print("\n===== COLUNAS PARA ANÁLISE DOS ATACANTES =====")

for coluna in df.columns:
    if any(
        termo.lower() in coluna.lower()
        for termo in [
            "Goal",
            "Gls",
            "xG",
            "Shot",
            "Sh",
            "SoT",
            "Touch",
            "Carr",
            "Prog",
            "Prg",
            "Penalty",
            "PK",
            "SCA",
            "GCA"
        ]
    ):
        print(coluna)

df_atacantes["Gls_90"] = (
    df_atacantes["Gls"] / df_atacantes["90s"]
)

df_atacantes["npxG_90"] = (
    df_atacantes["npxG"] / df_atacantes["90s"]
)

df_atacantes["PrgR_90"] = (
    df_atacantes["PrgR"] / df_atacantes["90s"]
)

df_atacantes["PrgC_90"] = (
    df_atacantes["PrgC"] / df_atacantes["90s"]
)

print("\n===== MÉTRICAS POR 90 DOS ATACANTES =====")

print(
    df_atacantes[
        [
            "Player",
            "Gls_90",
            "npxG_90",
            "PrgR_90",
            "PrgC_90"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

features_candidatas_atacantes = [
    "Gls_90",
    "npxG_90",
    "Sh/90",
    "SoT/90",
    "SoT%",
    "npxG/Sh",
    "G/Sh",
    "PrgR_90",
    "PrgC_90",
    "SCA90",
    "GCA90"
]

print("\n===== VALORES AUSENTES - ATACANTES =====")

print(
    df_atacantes[
        features_candidatas_atacantes
    ].isna().sum()
)

correlacao_atacantes = df_atacantes[
    features_candidatas_atacantes
].corr()

print("\n===== CORRELAÇÃO ENTRE FEATURES DOS ATACANTES =====")

print(
    correlacao_atacantes.round(2).to_string()
)

comparacao_gols_xg = df_atacantes[
    [
        "Player",
        "Squad",
        "Gls_90",
        "npxG_90"
    ]
].copy()

comparacao_gols_xg["Diferenca"] = (
    comparacao_gols_xg["Gls_90"]
    - comparacao_gols_xg["npxG_90"]
)

comparacao_gols_xg = comparacao_gols_xg.sort_values(
    by="Diferenca",
    ascending=False
)

print("\n===== MAIOR DIFERENÇA GOLS x npxG =====")

print(
    comparacao_gols_xg
    .head(10)
    .to_string(index=False)
)

print("\n===== MENOR DIFERENÇA GOLS x npxG =====")

print(
    comparacao_gols_xg
    .tail(10)
    .to_string(index=False)
)

comparacao_chutes = df_atacantes[
    [
        "Player",
        "Squad",
        "Sh/90",
        "SoT/90",
        "SoT%"
    ]
].copy()

comparacao_chutes = comparacao_chutes.sort_values(
    by="Sh/90",
    ascending=False
)

print("\n===== CHUTES DOS ATACANTES =====")

print(
    comparacao_chutes
    .head(15)
    .to_string(index=False)
)

comparacao_progressao_atacantes = df_atacantes[
    [
        "Player",
        "Squad",
        "PrgR_90",
        "PrgC_90"
    ]
].copy()

comparacao_progressao_atacantes = (
    comparacao_progressao_atacantes.sort_values(
        by="PrgR_90",
        ascending=False
    )
)

print("\n===== PROGRESSÃO DOS ATACANTES =====")

print(
    comparacao_progressao_atacantes
    .head(15)
    .to_string(index=False)
)

comparacao_qualidade_chutes = df_atacantes[
    [
        "Player",
        "Squad",
        "npxG/Sh",
        "G/Sh",
        "Gls_90",
        "Sh/90"
    ]
].copy()

comparacao_qualidade_chutes = (
    comparacao_qualidade_chutes.sort_values(
        by="npxG/Sh",
        ascending=False
    )
)

print("\n===== QUALIDADE DOS CHUTES =====")

print(
    comparacao_qualidade_chutes
    .head(15)
    .to_string(index=False)
)

features_atacantes = [
    "Gls_90",
    "npxG_90",
    "Sh/90",
    "SoT/90",
    "SoT%",
    "npxG/Sh",
    "G/Sh",
    "PrgR_90",
    "PrgC_90",
    "SCA90",
    "GCA90"
]

scaler_atacantes = StandardScaler()

features_atacantes_padronizadas = scaler_atacantes.fit_transform(
    df_atacantes[features_atacantes]
)

print("\n===== PADRONIZAÇÃO DOS ATACANTES =====")

print(
    "Formato da matriz:",
    features_atacantes_padronizadas.shape
)

print(
    "Quantidade de jogadores:",
    len(df_atacantes)
)

print(
    "Quantidade de features:",
    len(features_atacantes)
)

nome_referencia_atacante = "Kylian Mbappé"

atacantes_encontrados = df_atacantes[
    df_atacantes["Player"] == nome_referencia_atacante
]

print("\n===== ATACANTE DE REFERÊNCIA =====")

print(
    atacantes_encontrados[
        [
            "Player",
            "Squad",
            "Comp",
            "Pos",
            "Min"
        ]
    ].to_string(index=False)
)

print(
    "Quantidade de registros encontrados:",
    len(atacantes_encontrados)
)

if len(atacantes_encontrados) == 1:

    indice_referencia_atacante = atacantes_encontrados.index[0]

    posicao_referencia_atacante = df_atacantes.index.get_loc(
        indice_referencia_atacante
    )

    vetor_referencia_atacante = features_atacantes_padronizadas[
        posicao_referencia_atacante
    ]

    print("\n===== ATACANTE LOCALIZADO =====")

    print(
        "Índice no DataFrame:",
        indice_referencia_atacante
    )

    print(
        "Posição na matriz:",
        posicao_referencia_atacante
    )

    print(
        "Quantidade de features:",
        len(vetor_referencia_atacante)
    )

distancias_atacantes = np.linalg.norm(
    features_atacantes_padronizadas
    - vetor_referencia_atacante,
    axis=1
)

print("\n===== TESTE DAS DISTÂNCIAS - ATACANTES =====")

print(
    "Quantidade de distâncias:",
    len(distancias_atacantes)
)

print(
    "Distância do Mbappé para ele mesmo:",
    distancias_atacantes[posicao_referencia_atacante]
)

ranking_atacantes = df_atacantes.copy()

ranking_atacantes["Distancia"] = distancias_atacantes

ranking_atacantes = ranking_atacantes[
    ranking_atacantes["Player"] != nome_referencia_atacante
].copy()

ranking_atacantes = ranking_atacantes.sort_values(
    by="Distancia",
    ascending=True
)

print("\n===== TOP 10 ATACANTES SEMELHANTES A MBAPPÉ =====")

print(
    ranking_atacantes[
        [
            "Player",
            "Squad",
            "Pos",
            "Min",
            "Distancia"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

comparacao_mbappe_sane = pd.DataFrame({
    "Feature": features_atacantes,
    "Mbappe": df_atacantes.loc[
        indice_referencia_atacante,
        features_atacantes
    ].values,
    "Sane": ranking_atacantes.iloc[0][
        features_atacantes
    ].values
})

print("\n===== MBAPPÉ x SANÉ =====")

print(
    comparacao_mbappe_sane.to_string(index=False)
)

indice_sane = ranking_atacantes.iloc[0].name

posicao_sane = df_atacantes.index.get_loc(
    indice_sane
)

vetor_sane = features_atacantes_padronizadas[
    posicao_sane
]

diferencas = vetor_referencia_atacante - vetor_sane

quadrados = diferencas ** 2

soma_quadrados = np.sum(quadrados)

distancia_manual = np.sqrt(soma_quadrados)

print("\n===== VALIDAÇÃO MANUAL DA DISTÂNCIA =====")

print(
    "Soma dos quadrados:",
    soma_quadrados
)

print(
    "Raiz da soma:",
    distancia_manual
)

print(
    "Distância do ranking:",
    ranking_atacantes.iloc[0]["Distancia"]
)

features_atacantes_teste = [
    "Gls_90",
    "npxG_90",
    "Sh/90",
    "SoT%",
    "npxG/Sh",
    "G/Sh",
    "PrgR_90",
    "PrgC_90",
    "SCA90",
    "GCA90"
]

scaler_atacantes_teste = StandardScaler()

matriz_atacantes_teste = scaler_atacantes_teste.fit_transform(
    df_atacantes[features_atacantes_teste]
)

vetor_mbappe_teste = matriz_atacantes_teste[
    posicao_referencia_atacante
]

distancias_teste = np.linalg.norm(
    matriz_atacantes_teste - vetor_mbappe_teste,
    axis=1
)

ranking_teste = df_atacantes.copy()
ranking_teste["Distancia"] = distancias_teste

ranking_teste = ranking_teste[
    ranking_teste["Player"] != nome_referencia_atacante
].sort_values("Distancia")

print("\n===== TOP 10 SEM SoT/90 =====")

print(
    ranking_teste[
        ["Player", "Squad", "Distancia"]
    ]
    .head(10)
    .to_string(index=False)
)

top10_original = set(
    ranking_atacantes.head(10)["Player"]
)

top10_teste = set(
    ranking_teste.head(10)["Player"]
)

jogadores_em_comum = top10_original.intersection(
    top10_teste
)

print("\n===== ESTABILIDADE DO TOP 10 =====")

print(
    "Jogadores em comum:",
    len(jogadores_em_comum),
    "de 10"
)

print(
    "Percentual de permanência:",
    len(jogadores_em_comum) / 10 * 100,
    "%"
)

print(
    "Saiu do Top 10:",
    top10_original - top10_teste
)

print(
    "Entrou no Top 10:",
    top10_teste - top10_original
)

nome_teste_atacante = "Harry Kane"

jogador_teste = df_atacantes[
    df_atacantes["Player"] == nome_teste_atacante
]

print("\n===== SEGUNDO ATACANTE DE REFERÊNCIA =====")

print(
    jogador_teste[
        [
            "Player",
            "Squad",
            "Comp",
            "Pos",
            "Min"
        ]
    ].to_string(index=False)
)

print(
    "Quantidade de registros encontrados:",
    len(jogador_teste)
)

indice_kane = jogador_teste.index[0]

posicao_kane = df_atacantes.index.get_loc(
    indice_kane
)

vetor_kane = features_atacantes_padronizadas[
    posicao_kane
]

distancias_kane = np.linalg.norm(
    features_atacantes_padronizadas - vetor_kane,
    axis=1
)

ranking_kane = df_atacantes.copy()

ranking_kane["Distancia"] = distancias_kane

ranking_kane = ranking_kane[
    ranking_kane["Player"] != nome_teste_atacante
].copy()

ranking_kane = ranking_kane.sort_values(
    by="Distancia",
    ascending=True
)

print("\n===== TOP 10 ATACANTES SEMELHANTES A HARRY KANE =====")

print(
    ranking_kane[
        [
            "Player",
            "Squad",
            "Pos",
            "Min",
            "Distancia"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

top10_mbappe = set(
    ranking_atacantes.head(10)["Player"]
)

top10_kane = set(
    ranking_kane.head(10)["Player"]
)

comuns_mbappe_kane = top10_mbappe.intersection(
    top10_kane
)

print("\n===== MBAPPÉ x KANE =====")

print(
    "Jogadores em comum:",
    len(comuns_mbappe_kane),
    "de 10"
)

print(
    "Jogadores em comum:",
    comuns_mbappe_kane
)

def recomendar_atacantes(
    nome_jogador,
    quantidade=10,
    clube=None
):
    jogadores_encontrados = df_atacantes[
        df_atacantes["Player"] == nome_jogador
    ]

    if len(jogadores_encontrados) == 0:
        print(f"Jogador '{nome_jogador}' não encontrado.")
        return None

    if len(jogadores_encontrados) > 1:
        if clube is None:
            print(
                f"Existem {len(jogadores_encontrados)} registros "
                f"para '{nome_jogador}'. Escolha um clube:"
            )

            print(
                jogadores_encontrados[
                    ["Squad", "Comp", "Pos", "Min"]
                ].to_string(index=False)
            )

            return None

        jogadores_encontrados = jogadores_encontrados[
            jogadores_encontrados["Squad"] == clube
        ]

        if len(jogadores_encontrados) == 0:
            print(
                f"Não foi encontrado '{nome_jogador}' "
                f"no clube '{clube}'."
            )
            return None

    indice = jogadores_encontrados.index[0]

    posicao = df_atacantes.index.get_loc(indice)

    vetor = features_atacantes_padronizadas[
        posicao
    ]

    distancias = np.linalg.norm(
        features_atacantes_padronizadas - vetor,
        axis=1
    )

    resultado = df_atacantes.copy()

    resultado["Distancia"] = distancias

    resultado = resultado[
        resultado["Player"] != nome_jogador
    ].copy()

    resultado = resultado.sort_values(
        by="Distancia",
        ascending=True
    )

    return resultado.head(quantidade)

teste_funcao_atacantes = recomendar_atacantes(
    "Kylian Mbappé"
)

print(
    teste_funcao_atacantes[
        ["Player", "Squad", "Pos", "Distancia"]
    ].to_string(index=False)
)