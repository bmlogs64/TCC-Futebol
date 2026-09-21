import kagglehub
import pandas as pd
import os

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