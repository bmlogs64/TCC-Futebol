import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

caminho_atual = os.path.dirname(__file__)

caminho_csv = os.path.join(
    caminho_atual,
    "..",
    "data",
    "players_data-2024_2025_com_valores.csv"
)

df = pd.read_csv(caminho_csv)

MINUTOS_MINIMOS = 900

TAMANHO_POOL_TECNICO = 30

PESO_TECNICO = 0.60
PESO_FINANCEIRO = 0.40

df = df[
    df["Min"] >= MINUTOS_MINIMOS
].copy()


df_meias = df[
    df["Pos"].str.contains("MF", na=False)
].copy()

df_atacantes = df[
    df["Pos"].str.contains("FW", na=False)
].copy()

df_defensores = df[
    df["Pos"].str.contains("DF", na=False)
].copy()

df_goleiros = df[
    df["Pos"] == "GK"
].copy()


df_meias["xG_90"] = (
    df_meias["xG"] / df_meias["90s"]
)

df_meias["xAG_90"] = (
    df_meias["xAG"] / df_meias["90s"]
)

df_meias["KP_90"] = (
    df_meias["KP"] / df_meias["90s"]
)

df_meias["PPA_90"] = (
    df_meias["PPA"] / df_meias["90s"]
)

df_meias["PrgP_90"] = (
    df_meias["PrgP"] / df_meias["90s"]
)

df_meias["Touches_90"] = (
    df_meias["Touches"] / df_meias["90s"]
)

df_meias["PrgC_90"] = (
    df_meias["PrgC"] / df_meias["90s"]
)

df_meias["CPA_90"] = (
    df_meias["CPA"] / df_meias["90s"]
)

df_meias["Recov_90"] = (
    df_meias["Recov"] / df_meias["90s"]
)

features_meias = [
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

scaler_meias = StandardScaler()

features_meias_padronizadas = (
    scaler_meias.fit_transform(
        df_meias[features_meias]
    )
)

def recomendar_jogadores(
    nome_jogador,
    df_posicao,
    features_padronizadas,
    nome_modelo,
    quantidade=10,
    clube=None
):
    jogadores_encontrados = df_posicao[
        df_posicao["Player"] == nome_jogador
    ]

    if len(jogadores_encontrados) == 0:
        raise ValueError(
            f"Jogador '{nome_jogador}' não encontrado "
            f"no modelo de {nome_modelo}."
        )

    if len(jogadores_encontrados) > 1 and clube is None:
        clubes = jogadores_encontrados["Squad"].tolist()

        raise ValueError(
            f"Existem {len(jogadores_encontrados)} registros "
            f"para '{nome_jogador}'. "
            f"Escolha um clube: {', '.join(clubes)}."
        )

    if clube is not None:
        jogadores_encontrados = jogadores_encontrados[
            jogadores_encontrados["Squad"] == clube
        ]

    if len(jogadores_encontrados) == 0:
        raise ValueError(
            f"Não foi encontrado '{nome_jogador}' "
            f"no clube '{clube}'."
        )

    indice = jogadores_encontrados.index[0]

    posicao = df_posicao.index.get_loc(indice)

    vetor = features_padronizadas[posicao]

    valor_referencia = jogadores_encontrados.iloc[0][
    "market_value_in_eur"
]

    distancias = np.linalg.norm(
        features_padronizadas - vetor,
        axis=1
    )

    resultado = df_posicao.copy()

    resultado["Distancia"] = distancias

    numero_features = features_padronizadas.shape[1]

    resultado["DistanciaNormalizada"] = (
        resultado["Distancia"]
        / np.sqrt(numero_features)
    )

    resultado["ScoreTecnico"] = (
        1
        / (
            1
            + resultado["DistanciaNormalizada"]
        )
    )

    resultado["ValorReferencia"] = valor_referencia

    if (
        pd.notna(valor_referencia)
        and valor_referencia > 0
    ):
        resultado["EconomiaRelativa"] = (
            valor_referencia
            - resultado["market_value_in_eur"]
        ) / valor_referencia

        resultado["ScoreFinanceiro"] = (
            valor_referencia
            / (
                valor_referencia
                + resultado["market_value_in_eur"]
            )
        )

    else:
        resultado["EconomiaRelativa"] = np.nan
        resultado["ScoreFinanceiro"] = np.nan

    resultado = resultado[
        resultado["Player"] != nome_jogador
    ].copy()

    resultado = resultado.sort_values(
        by="Distancia",
        ascending=True
    )

    candidatos_tecnicos = resultado.head(
        TAMANHO_POOL_TECNICO
    ).copy()

    min_tecnico = candidatos_tecnicos[
        "ScoreTecnico"
    ].min()

    max_tecnico = candidatos_tecnicos[
        "ScoreTecnico"
    ].max()

    if max_tecnico > min_tecnico:
        candidatos_tecnicos[
            "ScoreTecnicoRelativo"
        ] = (
            candidatos_tecnicos["ScoreTecnico"]
            - min_tecnico
        ) / (
            max_tecnico
            - min_tecnico
        )
    else:
        candidatos_tecnicos[
            "ScoreTecnicoRelativo"
        ] = 1.0


    candidatos_tecnicos = candidatos_tecnicos[
        candidatos_tecnicos["ScoreTecnicoRelativo"] >= 0.50
    ].copy()

    candidatos_tecnicos = candidatos_tecnicos[
        candidatos_tecnicos["ScoreFinanceiro"].notna()
    ].copy()

    candidatos_tecnicos["ScoreTecnicoFinanceiro"] = (
        PESO_TECNICO
        * candidatos_tecnicos["ScoreTecnicoRelativo"]
        +
        PESO_FINANCEIRO
        * candidatos_tecnicos["ScoreFinanceiro"]
    )

    candidatos_tecnicos = candidatos_tecnicos.sort_values(
        by="ScoreTecnicoFinanceiro",
        ascending=False
    )

    return candidatos_tecnicos.head(
        quantidade
    )

def recomendar_meias(
    nome_jogador,
    quantidade=10,
    clube=None
):
    return recomendar_jogadores(
        nome_jogador=nome_jogador,
        df_posicao=df_meias,
        features_padronizadas=features_meias_padronizadas,
        nome_modelo="meio-campistas",
        quantidade=quantidade,
        clube=clube
    )

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

features_atacantes_padronizadas = (
    scaler_atacantes.fit_transform(
        df_atacantes[features_atacantes]
    )
)

def recomendar_atacantes(
    nome_jogador,
    quantidade=10,
    clube=None
):
    return recomendar_jogadores(
        nome_jogador=nome_jogador,
        df_posicao=df_atacantes,
        features_padronizadas=features_atacantes_padronizadas,
        nome_modelo="atacantes",
        quantidade=quantidade,
        clube=clube
    )

df_defensores["Tkl_90"] = (
    df_defensores["Tkl"] / df_defensores["90s"]
)

df_defensores["Int_90"] = (
    df_defensores["Int"] / df_defensores["90s"]
)

df_defensores["Blocks_90"] = (
    df_defensores["Blocks_stats_defense"]
    / df_defensores["90s"]
)

df_defensores["Clr_90"] = (
    df_defensores["Clr"] / df_defensores["90s"]
)

df_defensores["Won_90"] = (
    df_defensores["Won"] / df_defensores["90s"]
)

df_defensores["PrgP_90"] = (
    df_defensores["PrgP"] / df_defensores["90s"]
)

df_defensores["PrgC_90"] = (
    df_defensores["PrgC"] / df_defensores["90s"]
)

df_defensores["Crs_90"] = (
    df_defensores["Crs"] / df_defensores["90s"]
)

features_defensores = [
    "Tkl_90",
    "Tkl%",
    "Int_90",
    "Blocks_90",
    "Clr_90",
    "Won_90",
    "Won%",
    "PrgP_90",
    "PrgC_90",
    "Cmp%",
    "Crs_90"
]

scaler_defensores = StandardScaler()

features_defensores_padronizadas = (
    scaler_defensores.fit_transform(
        df_defensores[features_defensores]
    )
)

def recomendar_defensores(
    nome_jogador,
    quantidade=10,
    clube=None
):
    return recomendar_jogadores(
        nome_jogador=nome_jogador,
        df_posicao=df_defensores,
        features_padronizadas=features_defensores_padronizadas,
        nome_modelo="defensores",
        quantidade=quantidade,
        clube=clube
    )

df_goleiros["PSxG+/-_90"] = (
    df_goleiros["PSxG+/-"]
    / df_goleiros["90s"]
)

features_goleiros = [
    "GA90",
    "Save%",
    "CS%",
    "PSxG/SoT",
    "PSxG+/-_90",
    "Stp%",
    "#OPA/90",
    "Launch%"
]

scaler_goleiros = StandardScaler()

features_goleiros_padronizadas = (
    scaler_goleiros.fit_transform(
        df_goleiros[features_goleiros]
    )
)

def recomendar_goleiros(
    nome_jogador,
    quantidade=10,
    clube=None
):
    return recomendar_jogadores(
        nome_jogador=nome_jogador,
        df_posicao=df_goleiros,
        features_padronizadas=features_goleiros_padronizadas,
        nome_modelo="goleiros",
        quantidade=quantidade,
        clube=clube
    )

MODELOS = {
    "meio": recomendar_meias,
    "atacante": recomendar_atacantes,
    "defensor": recomendar_defensores,
    "goleiro": recomendar_goleiros
}

def recomendar(
    nome_jogador,
    modelo,
    quantidade=10,
    clube=None
):
    if modelo not in MODELOS:
        raise ValueError(
            f"Modelo '{modelo}' inválido. "
            f"Escolha entre: {', '.join(MODELOS.keys())}."
        )

    funcao_recomendacao = MODELOS[modelo]

    return funcao_recomendacao(
        nome_jogador=nome_jogador,
        quantidade=quantidade,
        clube=clube
    )

testes_finais = [
    ("Kylian Mbappé", "atacante"),
    ("Kevin De Bruyne", "meio"),
    ("Virgil van Dijk", "defensor"),
    ("Alisson", "goleiro")
]

for nome, modelo in testes_finais:

    print(
        f"\n=== RANKING FINAL - {nome.upper()} ==="
    )

    resultado = recomendar(
        nome_jogador=nome,
        modelo=modelo,
        quantidade=10
    )

    print(
        resultado[
            [
                "Player",
                "Squad",
                "market_value_in_eur",
                "ScoreTecnicoRelativo",
                "ScoreFinanceiro",
                "ScoreTecnicoFinanceiro"
            ]
        ]
        .to_string(index=False)
    )