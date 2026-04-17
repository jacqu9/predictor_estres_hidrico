import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Cargar datos
df = pd.read_csv("Crop_recommendation.csv")

# 2. Umbrales de estrés por tipo de cultivo
umbrales = {
    "rice":        {"lluvia": 150, "humedad": 70},
    "maize":       {"lluvia": 60,  "humedad": 55},
    "wheat":       {"lluvia": 50,  "humedad": 50},
    "mango":       {"lluvia": 80,  "humedad": 60},
    "cotton":      {"lluvia": 60,  "humedad": 55},
}
umbral_default = {"lluvia": 70, "humedad": 55}

def calcular_estres(row):
    cultivo = row["label"]
    umbral = umbrales.get(cultivo, umbral_default)
    return int(
        (row["rainfall"] < umbral["lluvia"]) &
        (row["humidity"] < umbral["humedad"])
    )

df["estres_hidrico"] = df.apply(calcular_estres, axis=1)

# 3. Variables
X = df[["temperature", "humidity", "rainfall", "ph"]]
y = df["estres_hidrico"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrenar 3 modelos
modelos = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Árbol de Decisión":  DecisionTreeClassifier(random_state=42),
    "Regresión Logística": LogisticRegression(max_iter=1000)
}

resultados = {}
mejor_nombre = None
mejor_score = 0
mejor_modelo = None

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    score = accuracy_score(y_test, modelo.predict(X_test))
    resultados[nombre] = round(score * 100, 2)
    print(f"{nombre}: {score*100:.2f}% precisión")
    if score > mejor_score:
        mejor_score = score
        mejor_nombre = nombre
        mejor_modelo = modelo

print(f"\n Mejor modelo: {mejor_nombre} con {mejor_score*100:.2f}%")

# 5. Guardar mejor modelo y resultados
with open("modelo_entrenado.pkl", "wb") as f:
    pickle.dump(mejor_modelo, f)

with open("resultados_modelos.pkl", "wb") as f:
    pickle.dump(resultados, f)

print(" Archivos guardados!")