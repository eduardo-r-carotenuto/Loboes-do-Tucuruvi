from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42

def numero(padrao, minimo, maximo):
    return {"tipo": "numero", "padrao": padrao, "min": minimo, "max": maximo}

def categoria(*opcoes):
    return {"tipo": "categoria", "padrao": opcoes[0], "opcoes": list(opcoes)}

PROBLEMAS = {
    "credito": {
        "titulo": "Risco de inadimplência",
        "tipo": "classificacao",
        "arquivo": "credito.csv",
        "alvo": "inadimplente",
        "resultado": "Probabilidade de inadimplência",
        "features": {
            "idade": numero(35, 18, 80),
            "renda_mensal": numero(5000, 1300, 60000),
            "tempo_emprego_anos": numero(5, 0, 40),
            "score_credito": numero(650, 300, 1000),
            "dividas_ativas": numero(1, 0, 15),
            "possui_imovel": categoria("nao", "sim"),
            "finalidade": categoria("pessoal", "veiculo", "reforma", "educacao", "negocio"),
            "valor_emprestimo": numero(20000, 1000, 200000),
            "prazo_meses": numero(24, 12, 60),
        },
    }
}
