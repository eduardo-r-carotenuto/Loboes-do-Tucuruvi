"""Parte 1: análise exploratória do problema de crédito."""
import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_DIR, OUTPUT_DIR

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "credito.csv")

    print("\n=== VISÃO GERAL ===")
    print(df.info())
    print("\nProporção de inadimplentes:")
    print(df["inadimplente"].value_counts(normalize=True).rename("proporcao"))
    print("\nValores ausentes:")
    print(df.isna().sum()[df.isna().sum() > 0])

    # Gráfico 1: proporção das classes
    taxas = df["inadimplente"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Adimplente", "Inadimplente"], taxas.values)
    ax.set_ylabel("Proporção")
    ax.set_title("Proporção de inadimplência")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_proporcao_inadimplencia.png", dpi=160)
    plt.close(fig)

    # Gráfico 2: inadimplência por faixa de score
    faixas = pd.cut(
        df["score_credito"],
        bins=[299, 500, 600, 700, 800, 1000],
        labels=["300-500", "501-600", "601-700", "701-800", "801-1000"]
    )
    por_score = df.assign(faixa_score=faixas).groupby(
        "faixa_score", observed=False
    )["inadimplente"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(por_score.index.astype(str), por_score.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_xlabel("Faixa de score de crédito")
    ax.set_title("Inadimplência por score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_inadimplencia_score.png", dpi=160)
    plt.close(fig)

    # Gráfico 3: inadimplência por finalidade
    por_finalidade = df.groupby("finalidade")["inadimplente"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(por_finalidade.index, por_finalidade.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_title("Inadimplência por finalidade do empréstimo")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_inadimplencia_finalidade.png", dpi=160)
    plt.close(fig)

    # Gráfico 4: inadimplência por posse de imóvel
    por_imovel = df.groupby("possui_imovel")["inadimplente"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(por_imovel.index, por_imovel.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_xlabel("Possui imóvel")
    ax.set_title("Inadimplência e posse de imóvel")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_inadimplencia_imovel.png", dpi=160)
    plt.close(fig)

    print("\nTaxa por faixa de score:\n", por_score)
    print("\nTaxa por finalidade:\n", por_finalidade)
    print("\nTaxa por posse de imóvel:\n", por_imovel)
    print("\nConclusão sobre descarte:")
    print("id_contrato deve ser descartado das features: é apenas um identificador e pode favorecer memorização, sem valor causal/preditivo útil para novos contratos.")

if __name__ == "__main__":
    main()
