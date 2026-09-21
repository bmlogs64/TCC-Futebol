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