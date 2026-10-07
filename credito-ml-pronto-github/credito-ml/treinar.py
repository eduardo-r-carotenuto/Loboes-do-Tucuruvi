"""Partes 2, 3 e 4: modelagem, avaliação e escolha de limiar."""
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, PROBLEMAS, RANDOM_STATE

CUSTO_INADIMPLENTE_APROVADO = 8000
CUSTO_BOM_RECUSADO = 1500

def criar_preprocessador(problema):
    numericas = [c for c, r in problema["features"].items() if r["tipo"] == "numero"]
    categoricas = [c for c, r in problema["features"].items() if r["tipo"] == "categoria"]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler())
        ]), numericas),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categoricas),
    ])

def pipeline(problema, modelo):
    return Pipeline([
        ("preprocessamento", criar_preprocessador(problema)),
        ("modelo", modelo)
    ])

def candidatos():
    return {
        "Regressão Logística": LogisticRegression(max_iter=1500),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=250, random_state=RANDOM_STATE
        ),
    }

def metricas_classificacao(y, prob, limiar=.5):
    pred = (prob >= limiar).astype(int)
    return {
        "acuracia": float(accuracy_score(y, pred)),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "matriz_confusao": confusion_matrix(y, pred).tolist(),
    }

def custo(y, prob, limiar):
    # prob >= limiar => recusar/alto risco; prob < limiar => aprovar
    recusar = prob >= limiar
    inadimplente_aprovado = int(((y == 1) & (~recusar)).sum())
    bom_recusado = int(((y == 0) & recusar).sum())
    total = (
        inadimplente_aprovado * CUSTO_INADIMPLENTE_APROVADO
        + bom_recusado * CUSTO_BOM_RECUSADO
    )
    return inadimplente_aprovado, bom_recusado, int(total)

def main():
    problema = PROBLEMAS["credito"]
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / problema["arquivo"])
    X = df[list(problema["features"])]
    y = df[problema["alvo"]]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE, stratify=y
    )

    # Parte 2 — comparação SEM usar o teste
    linhas_cv = []
    for nome, estimador in candidatos().items():
        p = pipeline(problema, estimador)
        cv = cross_validate(p, X_treino, y_treino, cv=5, scoring="roc_auc", n_jobs=-1)
        linhas_cv.append({
            "modelo": nome,
            "roc_auc_media": cv["test_score"].mean(),
            "roc_auc_desvio": cv["test_score"].std()
        })

    tabela_cv = pd.DataFrame(linhas_cv).sort_values("roc_auc_media", ascending=False)
    tabela_cv.to_csv(OUTPUT_DIR / "comparacao_modelos.csv", index=False)
    print("\n=== VALIDAÇÃO CRUZADA (treino) ===")
    print(tabela_cv.to_string(index=False))

    melhor_nome = tabela_cv.iloc[0]["modelo"]
    melhor = pipeline(problema, candidatos()[melhor_nome])

    # Parte 4 — probabilidades out-of-fold APENAS no treino
    prob_cv = cross_val_predict(
        melhor, X_treino, y_treino, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]

    limiares = np.arange(.10, .91, .05)
    analise = []
    for l in limiares:
        pred = (prob_cv >= l).astype(int)
        inad_aprov, bom_rec, custo_total = custo(y_treino.to_numpy(), prob_cv, l)
        analise.append({
            "limiar": round(float(l), 2),
            "precisao": precision_score(y_treino, pred, zero_division=0),
            "recall": recall_score(y_treino, pred, zero_division=0),
            "inadimplentes_aprovados": inad_aprov,
            "bons_recusados": bom_rec,
            "custo_validacao": custo_total
        })

    tabela_limiar = pd.DataFrame(analise)
    tabela_limiar.to_csv(OUTPUT_DIR / "analise_lim iares.csv".replace(" ", ""), index=False)
    melhor_limiar = float(tabela_limiar.loc[tabela_limiar["custo_validacao"].idxmin(), "limiar"])

    print("\n=== ANÁLISE DE LIMIAR (somente treino/CV) ===")
    print(tabela_limiar.to_string(index=False))
    print(f"\nLimiar recomendado: {melhor_limiar:.2f}")

    # Parte 3 — só agora o modelo vê o teste uma única vez
    melhor.fit(X_treino, y_treino)
    prob_teste = melhor.predict_proba(X_teste)[:, 1]

    teste_padrao = metricas_classificacao(y_teste, prob_teste, .5)
    teste_recomendado = metricas_classificacao(y_teste, prob_teste, melhor_limiar)
    inad_aprov, bom_rec, custo_teste = custo(y_teste.to_numpy(), prob_teste, melhor_limiar)

    fpr, tpr, _ = roc_curve(y_teste, prob_teste)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{melhor_nome} (AUC={teste_padrao['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC — conjunto de teste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_curva_roc.png", dpi=160)
    plt.close(fig)

    resultado = {
        "modelo_escolhido": melhor_nome,
        "validacao_cruzada": tabela_cv.to_dict(orient="records"),
        "teste_limiar_0_5": teste_padrao,
        "limiar_recomendado": melhor_limiar,
        "teste_limiar_recomendado": teste_recomendado,
        "custo_teste_limiar_recomendado": custo_teste,
        "inadimplentes_aprovados_teste": inad_aprov,
        "bons_recusados_teste": bom_rec,
        "interpretacao": {
            "falso_positivo": "Cliente bom classificado como alto risco e recusado: perda de oportunidade/lucro.",
            "falso_negativo": "Cliente inadimplente classificado como baixo risco e aprovado: prejuízo por inadimplência.",
            "erro_mais_caro": "Falso negativo, pois custa R$ 8.000 contra R$ 1.500 do falso positivo."
        }
    }

    (MODEL_DIR / "metricas.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(melhor, MODEL_DIR / "credito.joblib", compress=3)

    print("\n=== TESTE FINAL ===")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
