"""Interface web e API para previsão de risco de crédito."""
import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import DATA_DIR, MODEL_DIR, PROBLEMAS

if not (DATA_DIR / "credito.csv").exists():
    import gerar_dados
    gerar_dados.main()

if not (MODEL_DIR / "credito.joblib").exists():
    import treinar
    treinar.main()

problema = PROBLEMAS["credito"]
modelo = joblib.load(MODEL_DIR / "credito.joblib")
metricas = json.loads((MODEL_DIR / "metricas.json").read_text(encoding="utf-8"))

app = Flask(__name__)
app.json.ensure_ascii = False

def validar(dados):
    linha, erros = {}, []
    for campo, regra in problema["features"].items():
        valor = dados.get(campo)
        if valor in (None, ""):
            linha[campo] = None
        elif regra["tipo"] == "categoria":
            if valor not in regra["opcoes"]:
                erros.append(f"{campo}: escolha {', '.join(regra['opcoes'])}")
            linha[campo] = valor
        else:
            try:
                numero = float(valor)
                if not regra["min"] <= numero <= regra["max"]:
                    erros.append(f"{campo}: valor entre {regra['min']} e {regra['max']}")
                linha[campo] = numero
            except (TypeError, ValueError):
                erros.append(f"{campo}: deve ser numérico")
    return linha, erros

@app.get("/")
def index():
    return render_template("index.html", problema=problema, metricas=metricas)

@app.post("/api/prever")
def prever():
    linha, erros = validar(request.get_json(silent=True) or {})
    if erros:
        return jsonify({"erro": "; ".join(erros)}), 400

    X = pd.DataFrame([linha], columns=list(problema["features"]))
    prob = float(modelo.predict_proba(X)[0, 1])
    limiar = float(metricas["limiar_recomendado"])
    return jsonify({
        "probabilidade": prob,
        "limiar_recomendado": limiar,
        "classificacao": "alto risco" if prob >= limiar else "baixo risco"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
