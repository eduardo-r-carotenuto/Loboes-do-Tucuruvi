"""Gera o data/credito.csv sintético usado na atividade."""
import numpy as np
import pandas as pd
from config import DATA_DIR, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def gerar_credito(n=6000):
    df = pd.DataFrame({"id_contrato": [f"E{i:05d}" for i in range(1, n + 1)]})
    df["idade"] = np.clip(rng.normal(40, 12, n), 18, 80).round()
    df["renda_mensal"] = np.clip(4200 * rng.lognormal(0, .6, n), 1300, 60000).round(-1)
    df["tempo_emprego_anos"] = np.clip(rng.gamma(1.6, 3.5, n), 0, 40).round(1)
    df["score_credito"] = np.clip(rng.normal(640, 110, n), 300, 1000).round()
    df["dividas_ativas"] = rng.poisson(1.1, n)
    df["possui_imovel"] = rng.choice(["sim", "nao"], n, p=[.4, .6])
    df["finalidade"] = rng.choice(
        ["pessoal", "veiculo", "reforma", "educacao", "negocio"],
        n, p=[.35, .25, .15, .10, .15]
    )
    df["prazo_meses"] = rng.choice([12, 24, 36, 48, 60], n, p=[.15, .30, .25, .15, .15])
    df["valor_emprestimo"] = np.clip(
        df["renda_mensal"] * rng.uniform(.5, 6, n), 1000, 200000
    ).round(-2)

    comprometimento = (
        df["valor_emprestimo"] * 1.33 / df["prazo_meses"] / df["renda_mensal"]
    )
    z = (
        -2.2
        + 2.8 * np.clip(comprometimento, 0, 1.5)
        - .009 * (df["score_credito"] - 640)
        + .4 * df["dividas_ativas"]
        - .06 * df["tempo_emprego_anos"]
        - .35 * (df["possui_imovel"] == "sim")
        + .45 * (df["finalidade"] == "negocio")
        - .015 * (df["idade"] - 40)
        + rng.normal(0, .5, n)
    )
    df["inadimplente"] = (rng.random(n) < sigmoide(z)).astype(int)

    df.loc[rng.random(n) < .04, "renda_mensal"] = np.nan
    df.loc[rng.random(n) < .03, "tempo_emprego_anos"] = np.nan
    return df

def main():
    DATA_DIR.mkdir(exist_ok=True)
    df = gerar_credito()
    destino = DATA_DIR / "credito.csv"
    df.to_csv(destino, index=False)
    print(f"{destino}: {len(df)} linhas")
    print(f"Inadimplentes: {df['inadimplente'].mean():.2%}")

if __name__ == "__main__":
    main()
