import pandas as pd

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recomendacao import recomendar, df


app = FastAPI(
    title="Football Scouting API",
    description=(
        "API para identificação de jogadores tecnicamente "
        "semelhantes e financeiramente viáveis."
    ),
    version="1.0.0"
)

class JogadorReferencia(BaseModel):
    nome: str
    clube: str
    valor_mercado: str


class Criterios(BaseModel):
    peso_tecnico: str
    peso_financeiro: str
    corte_tecnico_relativo: str


class RecomendacaoJogador(BaseModel):
    posicao_ranking: int
    jogador: str
    clube: str
    idade: int | None
    valor_mercado: str
    vantagem_financeira: str
    similaridade_tecnica: str
    indice_recomendacao: str


class RespostaRecomendacao(BaseModel):
    jogador_referencia: JogadorReferencia
    criterios: Criterios
    total_recomendacoes: int
    recomendacoes: list[RecomendacaoJogador]

def formatar_euro(valor):
    if pd.isna(valor):
        return "Não disponível"

    valor = float(valor)

    sinal = "-" if valor < 0 else ""

    valor_formatado = (
        f"{abs(valor):,.0f}"
        .replace(",", ".")
    )

    return f"{sinal}€ {valor_formatado}"


def formatar_percentual(valor):
    if pd.isna(valor):
        return "Não disponível"

    return (
        f"{float(valor) * 100:.2f}%"
        .replace(".", ",")
    )

@app.get(
    "/recomendacoes",
    response_model=RespostaRecomendacao
)
def obter_recomendacoes(
    jogador: str,
    modelo: Literal[
        "meio",
        "atacante",
        "defensor",
        "goleiro"
    ],
    quantidade: int = Query(
        default=10,
        ge=1,
        le=10
    ),
    clube: str | None = None
):

    try:
        resultado = recomendar(
            nome_jogador=jogador,
            modelo=modelo,
            quantidade=quantidade,
            clube=clube
        )

    except ValueError as erro:
        raise HTTPException(
            status_code=400,
            detail=str(erro)
        )

    jogador_referencia = df[
        df["Player"] == jogador
    ].copy()

    if clube is not None:
        jogador_referencia = jogador_referencia[
            jogador_referencia["Squad"] == clube
        ]

    if jogador_referencia.empty:
        raise HTTPException(
            status_code=404,
            detail="Jogador de referência não encontrado."
        )

    referencia = jogador_referencia.iloc[0]

    valor_referencia = referencia[
        "market_value_in_eur"
    ]

    recomendacoes = []

    for posicao, (_, linha) in enumerate(
        resultado.iterrows(),
        start=1
    ):

        valor_candidato = linha[
            "market_value_in_eur"
        ]

        economia = (
            valor_referencia
            - valor_candidato
        )

        idade = None

        if pd.notna(linha["Age"]):
            idade = int(linha["Age"])


        recomendacoes.append({
            "posicao_ranking": posicao,

            "jogador":
                linha["Player"],

            "clube":
                linha["Squad"],

            "idade":
                idade,

            "valor_mercado":
                formatar_euro(
                    valor_candidato
                ),

            "vantagem_financeira":
                formatar_euro(
                    economia
                ),

            "similaridade_tecnica":
                formatar_percentual(
                    linha["ScoreTecnico"]
                ),

            "indice_recomendacao":
                formatar_percentual(
                    linha[
                        "ScoreTecnicoFinanceiro"
                    ]
                )
        })

    return {
        "jogador_referencia": {
            "nome":
                referencia["Player"],

            "clube":
                referencia["Squad"],

            "valor_mercado":
                formatar_euro(
                    valor_referencia
                )
        },

       "criterios": {
            "peso_tecnico": "60%",
            "peso_financeiro": "40%",
            "corte_tecnico_relativo": "50%"
        },

        "total_recomendacoes":
            len(recomendacoes),

        "recomendacoes":
            recomendacoes
    }

@app.get("/jogadores")
def buscar_jogadores(
    nome: str = Query(
        ...,
        min_length=2
    ),
    limite: int = Query(
        default=10,
        ge=1,
        le=20
    )
):
    jogadores = df[
        df["Player"].str.contains(
            nome,
            case=False,
            na=False,
            regex=False
        )
    ].copy()

    if jogadores.empty:
        raise HTTPException(
            status_code=404,
            detail="Nenhum jogador encontrado."
        )

    jogadores = jogadores[
        [
            "Player",
            "Squad",
            "Pos",
            "Age",
            "market_value_in_eur"
        ]
    ].drop_duplicates()

    jogadores = jogadores.head(
        limite
    )

    resultado = []

    for _, jogador in jogadores.iterrows():

        idade = None

        if pd.notna(jogador["Age"]):
            idade = int(jogador["Age"])

        resultado.append({
            "nome": jogador["Player"],
            "clube": jogador["Squad"],
            "posicao": jogador["Pos"],
            "idade": idade,
            "valor_mercado": formatar_euro(
                jogador["market_value_in_eur"]
            )
        })

    return {
        "busca": nome,
        "total_encontrados": len(resultado),
        "jogadores": resultado
    }