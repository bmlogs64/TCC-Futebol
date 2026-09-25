import os
import pandas as pd
import unicodedata
from difflib import SequenceMatcher

caminho_atual = os.path.dirname(__file__)

caminho_players = os.path.join(
    caminho_atual,
    "..",
    "data",
    "players.csv"
)

caminho_valores = os.path.join(
    caminho_atual,
    "..",
    "data",
    "player_valuations.csv"
)

caminho_estatisticas = os.path.join(
    caminho_atual,
    "..",
    "data",
    "players_data-2024_2025.csv"
)

players = pd.read_csv(caminho_players)

valores = pd.read_csv(caminho_valores)

estatisticas = pd.read_csv(caminho_estatisticas)

valores["date"] = pd.to_datetime(
    valores["date"],
    errors="coerce"
)


DATA_LIMITE = pd.Timestamp("2025-06-30")


valores = valores[
    valores["date"] <= DATA_LIMITE
].copy()


valores = valores.sort_values(
    by=["player_id", "date"]
)


ultimos_valores = (
    valores
    .groupby("player_id")
    .tail(1)
    .copy()
)


base_valores = ultimos_valores.merge(
    players[
    [
        "player_id",
        "name",
        "first_name",
        "last_name",
        "player_code",
        "position",
        "date_of_birth"
    ]
],
    on="player_id",
    how="left"
)


base_valores = base_valores[
    [
        "player_id",
        "name",
        "first_name",
        "last_name",
        "player_code",
        "position",
        "date_of_birth",
        "current_club_name",
        "market_value_in_eur",
        "date"
    ]
]

base_valores = base_valores.rename(
    columns={
        "date": "valuation_date"
    }
)

def normalizar_nome(nome):
    if pd.isna(nome):
        return ""

    nome = str(nome).lower().strip()

    nome = unicodedata.normalize(
        "NFKD",
        nome
    )

    nome = "".join(
        caractere
        for caractere in nome
        if not unicodedata.combining(caractere)
    )

    return nome


base_valores["nome_normalizado"] = (
    base_valores["name"]
    .apply(normalizar_nome)
)

estatisticas["nome_normalizado"] = (
    estatisticas["Player"]
    .apply(normalizar_nome)
)

nomes_transfermarkt = set(
    base_valores["nome_normalizado"]
)

estatisticas["tem_correspondencia_nome"] = (
    estatisticas["nome_normalizado"]
    .isin(nomes_transfermarkt)
)

percentual = (
    estatisticas["tem_correspondencia_nome"].mean()
    * 100
)

contagem_candidatos = (
    base_valores
    .groupby("nome_normalizado")["player_id"]
    .nunique()
)


estatisticas["quantidade_candidatos_tm"] = (
    estatisticas["nome_normalizado"]
    .map(contagem_candidatos)
    .fillna(0)
    .astype(int)
)


sem_candidato = (
    estatisticas["quantidade_candidatos_tm"] == 0
).sum()

candidato_unico = (
    estatisticas["quantidade_candidatos_tm"] == 1
).sum()

multiplos_candidatos = (
    estatisticas["quantidade_candidatos_tm"] > 1
).sum()

ambiguos = estatisticas[
    estatisticas["quantidade_candidatos_tm"] > 1
][
    [
        "Player",
        "Squad",
        "Pos",
        "quantidade_candidatos_tm"
    ]
].drop_duplicates()

def posicoes_fbref_para_transfermarkt(posicao_fbref):
    mapa = {
        "GK": "Goalkeeper",
        "DF": "Defender",
        "MF": "Midfield",
        "FW": "Attack"
    }

    posicoes = str(posicao_fbref).split(",")

    return {
        mapa[posicao.strip()]
        for posicao in posicoes
        if posicao.strip() in mapa
    }


def contar_candidatos_por_posicao(linha):
    candidatos = base_valores[
        base_valores["nome_normalizado"]
        == linha["nome_normalizado"]
    ]

    if len(candidatos) == 0:
        return 0

    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos = candidatos[
        candidatos["position"].isin(
            posicoes_validas
        )
    ]

    return candidatos["player_id"].nunique()


estatisticas["candidatos_apos_posicao"] = (
    estatisticas.apply(
        contar_candidatos_por_posicao,
        axis=1
    )
)

ambiguos_posicao = estatisticas[
    estatisticas["candidatos_apos_posicao"] > 1
][
    [
        "Player",
        "Squad",
        "Pos",
        "candidatos_apos_posicao"
    ]
].drop_duplicates()

def normalizar_clube(nome):
    if pd.isna(nome):
        return ""

    nome = normalizar_nome(nome)

    termos_remover = [
        " football club",
        " futebol clube",
        " futbol club",
        " fc",
        " cf",
        " afc",
        " calcio",
        " ac",
        " sc",
        " ss",
        " as",
        " sv",
        " rc",
        " rcd"
    ]

    for termo in termos_remover:
        nome = nome.replace(termo, "")

    nome = nome.replace("&", " ")

    nome = " ".join(
        nome.split()
    )

    return nome.strip()

def similaridade_clube(a, b):
    clube_a = normalizar_clube(a)
    clube_b = normalizar_clube(b)

    if not clube_a or not clube_b:
        return 0

    if clube_a in clube_b or clube_b in clube_a:
        return 1.0

    return SequenceMatcher(
        None,
        clube_a,
        clube_b
    ).ratio()

def encontrar_por_clube(linha):
    candidatos = base_valores[
        base_valores["nome_normalizado"]
        == linha["nome_normalizado"]
    ].copy()

    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos = candidatos[
        candidatos["position"].isin(
            posicoes_validas
        )
    ].copy()

    if len(candidatos) <= 1:
        return None

    candidatos["similaridade_clube"] = (
        candidatos["current_club_name"]
        .apply(
            lambda clube:
            similaridade_clube(
                linha["Squad"],
                clube
            )
        )
    )

    melhor = candidatos.sort_values(
        by="similaridade_clube",
        ascending=False
    ).iloc[0]

    return pd.Series({
        "Player": linha["Player"],
        "Squad_FBref": linha["Squad"],
        "Pos": linha["Pos"],
        "nome_transfermarkt": melhor["name"],
        "clube_transfermarkt": melhor["current_club_name"],
        "player_id": melhor["player_id"],
        "similaridade_clube": melhor["similaridade_clube"]
    })

resultados_clube = []

for _, linha in estatisticas[
    estatisticas["candidatos_apos_posicao"] > 1
].iterrows():

    resultado = encontrar_por_clube(linha)

    if resultado is not None:
        resultados_clube.append(resultado)


df_resultados_clube = pd.DataFrame(
    resultados_clube
)

casos_baixos = df_resultados_clube[
    df_resultados_clube["similaridade_clube"] < 0.70
]

base_valores["date_of_birth"] = pd.to_datetime(
    base_valores["date_of_birth"],
    errors="coerce"
)

DATA_REFERENCIA_IDADE = pd.Timestamp("2024-08-01")


def calcular_idade(data_nascimento):
    if pd.isna(data_nascimento):
        return None

    idade = (
        DATA_REFERENCIA_IDADE.year
        - data_nascimento.year
    )

    if (
        DATA_REFERENCIA_IDADE.month,
        DATA_REFERENCIA_IDADE.day
    ) < (
        data_nascimento.month,
        data_nascimento.day
    ):
        idade -= 1

    return idade

base_valores["first_name_normalizado"] = (
    base_valores["first_name"]
    .apply(normalizar_nome)
)

base_valores["nome_completo_normalizado"] = (
    (
        base_valores["first_name"].fillna("")
        + " "
        + base_valores["last_name"].fillna("")
    )
    .apply(normalizar_nome)
)

base_valores["player_code_normalizado"] = (
    base_valores["player_code"]
    .fillna("")
    .str.replace("-", " ", regex=False)
    .apply(normalizar_nome)
)

base_valores["idade_transfermarkt"] = (
    base_valores["date_of_birth"]
    .apply(calcular_idade)
)

def contar_candidatos_nome_idade_sem_posicao(linha):
    if not (
        linha["quantidade_candidatos_tm"] > 0
        and linha["candidatos_apos_posicao"] == 0
    ):
        return 0

    if pd.isna(linha["Age"]):
        return 0

    candidatos = base_valores[
        base_valores["nome_normalizado"]
        == linha["nome_normalizado"]
    ].copy()

    candidatos["diferenca_idade"] = (
        candidatos["idade_transfermarkt"]
        - linha["Age"]
    ).abs()

    candidatos = candidatos[
        candidatos["diferenca_idade"] <= 1
    ]

    return candidatos["player_id"].nunique()

estatisticas["candidatos_nome_idade_sem_posicao"] = (
    estatisticas.apply(
        contar_candidatos_nome_idade_sem_posicao,
        axis=1
    )
)

def resolver_candidatos(
    candidatos,
    linha,
    metodo
):
    if len(candidatos) == 0:
        return None

    candidatos = candidatos.copy()

    if pd.notna(linha["Age"]):
        candidatos["diferenca_idade"] = (
            candidatos["idade_transfermarkt"]
            - linha["Age"]
        ).abs()

        candidatos_idade = candidatos[
            candidatos["diferenca_idade"] <= 1
        ].copy()

        if len(candidatos_idade) == 1:
            jogador = candidatos_idade.iloc[0]

            return jogador, f"{metodo}_idade"

        if len(candidatos_idade) > 1:
            menor_diferenca = (
                candidatos_idade["diferenca_idade"].min()
            )

            melhores_idade = candidatos_idade[
                candidatos_idade["diferenca_idade"]
                == menor_diferenca
            ].copy()

            if len(melhores_idade) == 1:
                jogador = melhores_idade.iloc[0]

                return jogador, f"{metodo}_idade"

            candidatos = melhores_idade

    candidatos["similaridade_clube"] = (
        candidatos["current_club_name"]
        .apply(
            lambda clube:
            similaridade_clube(
                linha["Squad"],
                clube
            )
        )
    )

    melhor_similaridade = (
        candidatos["similaridade_clube"].max()
    )

    melhores_clube = candidatos[
        candidatos["similaridade_clube"]
        == melhor_similaridade
    ]

    if (
        len(melhores_clube) == 1
        and melhor_similaridade >= 0.70
    ):
        jogador = melhores_clube.iloc[0]

        return jogador, f"{metodo}_clube"

    return None

def associar_transfermarkt(linha):
    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos_exatos = base_valores[
        (
            base_valores["nome_normalizado"]
            == linha["nome_normalizado"]
        )
        &
        (
            base_valores["position"]
            .isin(posicoes_validas)
        )
    ].copy()

    if len(candidatos_exatos) == 1:
        jogador = candidatos_exatos.iloc[0]
        metodo = "nome_posicao"

    else:
        resultado = resolver_candidatos(
            candidatos_exatos,
            linha,
            "nome"
        )

        if resultado is not None:
            jogador, metodo = resultado

        else:
            candidatos_sem_posicao = base_valores[
                base_valores["nome_normalizado"]
                == linha["nome_normalizado"]
            ].copy()

            resultado = resolver_candidatos(
                candidatos_sem_posicao,
                linha,
                "nome_sem_posicao"
            )

            if resultado is not None:
                jogador, metodo = resultado

            else:
                candidatos_abreviados = base_valores[
                    (
                        base_valores[
                            "first_name_normalizado"
                        ]
                        == linha["nome_normalizado"]
                    )
                    &
                    (
                        base_valores["position"]
                        .isin(posicoes_validas)
                    )
                ].copy()

                resultado = resolver_candidatos(
                    candidatos_abreviados,
                    linha,
                    "nome_abreviado"
                )

                if resultado is None:
                    return pd.Series({
                        "tm_player_id": None,
                        "tm_name": None,
                        "tm_club": None,
                        "market_value_in_eur": None,
                        "valuation_date": None,
                        "metodo_correspondencia":
                            "nao_resolvido"
                    })

                jogador, metodo = resultado

    return pd.Series({
        "tm_player_id": jogador["player_id"],
        "tm_name": jogador["name"],
        "tm_club": jogador["current_club_name"],
        "market_value_in_eur":
            jogador["market_value_in_eur"],
        "valuation_date":
            jogador["valuation_date"],
        "metodo_correspondencia": metodo
    })

caminho_cache = os.path.join(
    caminho_atual,
    "..",
    "data",
    "estatisticas_com_valores_temp.csv"
)

RECALCULAR_ASSOCIACOES = False

if RECALCULAR_ASSOCIACOES:
    associacoes = estatisticas.apply(
        associar_transfermarkt,
        axis=1
    )

    estatisticas_com_valores = pd.concat(
        [
            estatisticas,
            associacoes
        ],
        axis=1
    )

    estatisticas_com_valores.to_csv(
        caminho_cache,
        index=False
    )

else:
    estatisticas_com_valores = pd.read_csv(
        caminho_cache
    )

def contar_candidatos_nome_abreviado(linha):
    if linha["candidatos_apos_posicao"] == 1:
        return 0

    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos = base_valores[
        (
            base_valores["first_name_normalizado"]
            == linha["nome_normalizado"]
        )
        &
        (
            base_valores["position"]
            .isin(posicoes_validas)
        )
    ].copy()

    candidatos["diferenca_idade"] = (
        candidatos["idade_transfermarkt"]
        - linha["Age"]
    ).abs()

    candidatos = candidatos[
        candidatos["diferenca_idade"] <= 1
    ]

    return candidatos["player_id"].nunique()

estatisticas["candidatos_nome_abreviado"] = (
    estatisticas.apply(
        contar_candidatos_nome_abreviado,
        axis=1
    )
)

jogadores_baixa_similaridade = casos_baixos["Player"].tolist()

for _, linha in estatisticas[
    estatisticas["Player"].isin(
        jogadores_baixa_similaridade
    )
].iterrows():

    candidatos = base_valores[
        base_valores["nome_normalizado"]
        == linha["nome_normalizado"]
    ].copy()

    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos = candidatos[
        candidatos["position"].isin(
            posicoes_validas
        )
    ].copy()

    candidatos["idade_transfermarkt"] = (
        candidatos["date_of_birth"]
        .apply(calcular_idade)
    )

    candidatos["diferenca_idade"] = (
        candidatos["idade_transfermarkt"]
        - linha["Age"]
    ).abs()

ids_teste = [
    476344,
    204072
]

linha_henrique = estatisticas[
    (estatisticas["Player"] == "Henrique")
    &
    (estatisticas["Squad"] == "Valladolid")
].iloc[0]

posicoes_validas = posicoes_fbref_para_transfermarkt(
    linha_henrique["Pos"]
)

candidatos_henrique = base_valores[
    (
        base_valores["first_name_normalizado"]
        == linha_henrique["nome_normalizado"]
    )
    &
    (
        base_valores["position"].isin(
            posicoes_validas
        )
    )
].copy()

candidatos_henrique["diferenca_idade"] = (
    candidatos_henrique["idade_transfermarkt"]
    - linha_henrique["Age"]
).abs()

candidatos_henrique = candidatos_henrique[
    candidatos_henrique["diferenca_idade"] <= 1
]

resolvidos = (
    estatisticas_com_valores[
        "tm_player_id"
    ].notna().sum()
)

nao_resolvidos = (
    estatisticas_com_valores[
        "tm_player_id"
    ].isna().sum()
)

percentual = (
    resolvidos
    / len(estatisticas_com_valores)
    * 100
)

nao_resolvidos_df = estatisticas_com_valores[
    estatisticas_com_valores["metodo_correspondencia"]
    == "nao_resolvido"
].copy()

sem_nome = nao_resolvidos_df[
    nao_resolvidos_df["quantidade_candidatos_tm"] == 0
]

nome_sem_posicao = nao_resolvidos_df[
    (
        nao_resolvidos_df["quantidade_candidatos_tm"] > 0
    )
    &
    (
        nao_resolvidos_df["candidatos_apos_posicao"] == 0
    )
]

ambiguos_restantes = nao_resolvidos_df[
    nao_resolvidos_df["candidatos_apos_posicao"] > 1
]

grupo_posicao_incompativel = estatisticas[
    (
        estatisticas["quantidade_candidatos_tm"] > 0
    )
    &
    (
        estatisticas["candidatos_apos_posicao"] == 0
    )
]

def similaridade_nome(nome_fbref, candidato):
    nome_fbref = normalizar_nome(nome_fbref)

    nomes_candidato = [
        candidato["nome_normalizado"],
        candidato["nome_completo_normalizado"],
        candidato["player_code_normalizado"]
    ]

    melhores_similaridades = []

    for nome_tm in nomes_candidato:
        if not nome_tm:
            continue

        if nome_fbref == nome_tm:
            melhores_similaridades.append(1.0)
            continue

        similaridade_caracteres = SequenceMatcher(
            None,
            nome_fbref,
            nome_tm
        ).ratio()

        tokens_fbref = set(
            nome_fbref.split()
        )

        tokens_tm = set(
            nome_tm.split()
        )

        if tokens_fbref and tokens_tm:
            intersecao = (
                tokens_fbref
                & tokens_tm
            )

            similaridade_tokens = (
                2 * len(intersecao)
                / (
                    len(tokens_fbref)
                    + len(tokens_tm)
                )
            )
        else:
            similaridade_tokens = 0

        melhores_similaridades.append(
            max(
                similaridade_caracteres,
                similaridade_tokens
            )
        )

    if not melhores_similaridades:
        return 0

    return max(melhores_similaridades)

def melhor_candidato_nome_diferente(linha):
    posicoes_validas = posicoes_fbref_para_transfermarkt(
        linha["Pos"]
    )

    candidatos = base_valores.copy()

    if pd.notna(linha["Age"]):
        candidatos["diferenca_idade"] = (
            candidatos["idade_transfermarkt"]
            - linha["Age"]
        ).abs()

        candidatos = candidatos[
            candidatos["diferenca_idade"] <= 1
        ].copy()

    candidatos_posicao = candidatos[
        candidatos["position"].isin(
            posicoes_validas
        )
    ].copy()

    if len(candidatos_posicao) > 0:
        candidatos = candidatos_posicao

    candidatos["similaridade_nome"] = (
        candidatos.apply(
            lambda candidato:
            similaridade_nome(
                linha["Player"],
                candidato
            ),
            axis=1
        )
    )

    candidatos["similaridade_clube"] = (
        candidatos["current_club_name"]
        .apply(
            lambda clube:
            similaridade_clube(
                linha["Squad"],
                clube
            )
        )
    )

    candidatos = candidatos.sort_values(
        by=[
            "similaridade_nome",
            "similaridade_clube"
        ],
        ascending=False
    )

    if len(candidatos) == 0:
        return None

    melhor = candidatos.iloc[0]

    if pd.notna(linha["Age"]):
        diferenca_idade_melhor = abs(
            melhor["idade_transfermarkt"]
            - linha["Age"]
        )
    else:
        diferenca_idade_melhor = None


    posicao_compativel = (
        melhor["position"]
        in posicoes_validas
    )

    if len(candidatos) > 1:
        segundo = candidatos.iloc[1]

        segundo_nome = segundo["name"]
        segunda_similaridade = segundo[
            "similaridade_nome"
        ]
    else:
        segundo_nome = None
        segunda_similaridade = 0

    margem = (
        melhor["similaridade_nome"]
        - segunda_similaridade
    )

    return pd.Series({
    "Player": linha["Player"],
    "Squad_FBref": linha["Squad"],
    "Pos_FBref": linha["Pos"],
    "Age_FBref": linha["Age"],

    "tm_player_id": melhor["player_id"],
    "tm_name": melhor["name"],
    "tm_position": melhor["position"],
    "tm_club": melhor["current_club_name"],
    "tm_age": melhor["idade_transfermarkt"],

    "market_value_in_eur":melhor["market_value_in_eur"],

    "valuation_date":melhor["valuation_date"],

    "diferenca_idade":
        diferenca_idade_melhor,

    "posicao_compativel":
        posicao_compativel,

    "similaridade_nome":
        melhor["similaridade_nome"],

    "similaridade_clube":
        melhor["similaridade_clube"],

    "segundo_candidato":
        segundo_nome,

    "segunda_similaridade":
        segunda_similaridade,

    "margem_similaridade":
        margem
})

caminho_fuzzy = os.path.join(
    caminho_atual,
    "..",
    "data",
    "fuzzy_temp.csv"
)

RECALCULAR_FUZZY = False

if RECALCULAR_FUZZY:
    nao_resolvidos_atual = estatisticas_com_valores[
        estatisticas_com_valores[
            "metodo_correspondencia"
        ] == "nao_resolvido"
    ]

    resultados_fuzzy = []

    for _, linha in nao_resolvidos_atual.iterrows():
        resultado = melhor_candidato_nome_diferente(
            linha
        )

        if resultado is not None:
            resultados_fuzzy.append(resultado)

    df_fuzzy = pd.DataFrame(
        resultados_fuzzy
    )

    df_fuzzy.to_csv(
        caminho_fuzzy,
        index=False
    )

else:
    df_fuzzy = pd.read_csv(
        caminho_fuzzy
    )

alta_confianca = df_fuzzy[
    (
        df_fuzzy["similaridade_nome"] >= 0.90
    )
    &
    (
        df_fuzzy["margem_similaridade"] >= 0.05
    )
    &
    (
        df_fuzzy["diferenca_idade"] <= 1
    )
    &
    (
        df_fuzzy["posicao_compativel"] == True
    )
]

estatisticas_final = estatisticas_com_valores.copy()

incorporados_fuzzy = 0
conflitos_fuzzy = 0

for _, jogador in alta_confianca.iterrows():

    mascara = (
        (estatisticas_final["Player"] == jogador["Player"])
        &
        (estatisticas_final["Squad"] == jogador["Squad_FBref"])
        &
        (estatisticas_final["Age"] == jogador["Age_FBref"])
    )

    quantidade_encontrada = mascara.sum()

    if quantidade_encontrada == 1:

        estatisticas_final.loc[
            mascara,
            "tm_player_id"
        ] = jogador["tm_player_id"]

        estatisticas_final.loc[
            mascara,
            "tm_name"
        ] = jogador["tm_name"]

        estatisticas_final.loc[
            mascara,
            "tm_club"
        ] = jogador["tm_club"]

        estatisticas_final.loc[
            mascara,
            "market_value_in_eur"
        ] = jogador["market_value_in_eur"]

        estatisticas_final.loc[
            mascara,
            "valuation_date"
        ] = jogador["valuation_date"]

        estatisticas_final.loc[
            mascara,
            "metodo_correspondencia"
        ] = "fuzzy_alta_confianca"

        incorporados_fuzzy += 1

    else:
        conflitos_fuzzy += 1

print("\n=== CANDIDATOS FUZZY DE ALTA CONFIANÇA ===")

print(
    "Quantidade:",
    len(alta_confianca)
)

print(
    alta_confianca[
        [
            "Player",
            "Squad_FBref",
            "tm_name",
            "tm_club",
            "similaridade_nome",
            "margem_similaridade",
            "diferenca_idade",
            "similaridade_clube"
        ]
    ]
    .sort_values(
        by="similaridade_nome",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\n=== CASOS MAIS FRACOS DA ALTA CONFIANÇA ===")

print(
    alta_confianca[
        [
            "Player",
            "Squad_FBref",
            "tm_name",
            "tm_club",
            "similaridade_nome",
            "margem_similaridade",
            "diferenca_idade",
            "similaridade_clube"
        ]
    ]
    .sort_values(
        by=[
            "similaridade_nome",
            "margem_similaridade"
        ],
        ascending=True
    )
    .head(25)
    .to_string(index=False)
)

print("\n=== VALORES DOS FUZZY DE ALTA CONFIANÇA ===")

print(
    alta_confianca[
        [
            "Player",
            "tm_name",
            "market_value_in_eur",
            "valuation_date"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\n=== INCORPORAÇÃO FUZZY ===")

print(
    "Incorporados:",
    incorporados_fuzzy
)

print(
    "Conflitos:",
    conflitos_fuzzy
)

resolvidos_final = (
    estatisticas_final[
        "tm_player_id"
    ].notna().sum()
)

nao_resolvidos_final = (
    estatisticas_final[
        "tm_player_id"
    ].isna().sum()
)

cobertura_final = (
    resolvidos_final
    / len(estatisticas_final)
    * 100
)

print(
    "Resolvidos:",
    resolvidos_final
)

print(
    "Não resolvidos:",
    nao_resolvidos_final
)

print(
    f"Cobertura: {cobertura_final:.2f}%"
)

caminho_saida_final = os.path.join(
    caminho_atual,
    "..",
    "data",
    "players_data-2024_2025_com_valores.csv"
)

estatisticas_final.to_csv(
    caminho_saida_final,
    index=False
)

print(
    "\nBase final salva em:",
    caminho_saida_final
)