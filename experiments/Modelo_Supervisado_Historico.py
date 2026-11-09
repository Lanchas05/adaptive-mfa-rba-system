import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==============================================================================
# 1. CARGA DE MUESTRA DEL RBA DATASET (INTACTO)
# ==============================================================================
FILE_PATH = "/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv"

print("Cargando muestra del dataset...")
df = pd.read_csv(FILE_PATH, nrows=50000)

df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"], errors="coerce")
df = df.sort_values(by=["User ID", "Login Timestamp"]).reset_index(drop=True)

df["login_hour"] = df["Login Timestamp"].dt.hour.fillna(12)
df["RTT_clean"] = df["Round-Trip Time [ms]"].fillna(
    df["Round-Trip Time [ms]"].median()
)
df["Login_Success_num"] = df["Login Successful"].astype(int)

cat_cols = [
    "Country",
    "ASN",
    "Browser Name and Version",
    "OS Name and Version",
    "Device Type",
]
for col in cat_cols:
  df[col] = df[col].astype(str)
  freq = df[col].value_counts(normalize=True)
  df[col + "_freq"] = df[col].map(freq)

# ==============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS POR USUARIO (INTACTO)
# ==============================================================================
print("Calculando histórico y variables relativas por usuario...")

df["user_rtt_mean_prev"] = df.groupby("User ID")["RTT_clean"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df["user_delta_rtt"] = (
    (df["RTT_clean"] - df["user_rtt_mean_prev"]).abs().fillna(0)
)


def flag_new_entity(series):
  seen = set()
  flags = []
  for item in series:
    if not seen:
      flags.append(0)
    else:
      flags.append(1 if item not in seen else 0)
    seen.add(item)
  return flags


df["is_new_country_user"] = df.groupby("User ID")["Country"].transform(
    flag_new_entity
)
df["is_new_asn_user"] = df.groupby("User ID")["ASN"].transform(flag_new_entity)

df["device_fingerprint"] = (
    df["Browser Name and Version"]
    + "_"
    + df["OS Name and Version"]
    + "_"
    + df["Device Type"]
)
df["is_new_device_user"] = df.groupby("User ID")[
    "device_fingerprint"
].transform(flag_new_entity)

df["prev_failed_login"] = (
    df.groupby("User ID")["Login_Success_num"]
    .shift(1)
    .apply(lambda x: 1 if x == 0 else 0)
    .fillna(0)
)

# ==============================================================================
# 3. PREPARACIÓN SUPERVISADA: DIVISIÓN TRAIN / TEST (80% / 20%)
# ==============================================================================
features = [
    "login_hour",
    "RTT_clean",
    "Login_Success_num",
    "Country_freq",
    "ASN_freq",
    "Browser Name and Version_freq",
    "OS Name and Version_freq",
    "Device Type_freq",
    "user_delta_rtt",
    "is_new_country_user",
    "is_new_asn_user",
    "is_new_device_user",
    "prev_failed_login",
]

X = df[features]
y = ((df["Is Attack IP"] == True) | (df["Is Account Takeover"] == True)).astype(
    int
)

# Dividimos: 80% para enseñar al modelo (Train) y 20% oculto para examinarlo (Test)
# stratify=y asegura que haya el mismo % de ataques (9%) tanto en Train como en Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# IMPORTANTE EN SUPERVISADO: El escalador aprende SOLO de Train (fit_transform)
# y se aplica a Test (transform) para no filtrar información del examen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

total_ataques_test = int(y_test.sum())
print(
    f"Filas para Entrenar (80%): {len(X_train)} | Filas para Examen Test (20%):"
    f" {len(X_test)}"
)
print(
    "Eventos maliciosos reales dentro del Examen Test:"
    f" {total_ataques_test}\n"
)

# ==============================================================================
# 4. DEFINICIÓN DE LOS 3 ALGORITMOS SUPERVISADOS
# ==============================================================================
# class_weight='balanced' obliga al modelo a prestar mucha atención a los ataques
# aunque solo sean el 9% de los datos.
modelos_supervisados = {
    "1. Regresión Logística (Baseline)": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "2. Random Forest (200 árboles)": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "3. HistGradientBoosting (Boosting)": HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=200, random_state=42
    ),
}

# ==============================================================================
# 5. ENTRENAMIENTO (.fit) Y EVALUACIÓN EN EL CONJUNTO DE TEST (.predict)
# ==============================================================================
resultados = []

for nombre, modelo in modelos_supervisados.items():
  print(f"Entrenando y evaluando {nombre}...")

  # 1. El modelo estudia con los datos de entrenamiento Y SUS ETIQUETAS (y_train)
  modelo.fit(X_train_scaled, y_train)

  # 2. El modelo hace el examen sobre datos que NUNCA ha visto (X_test_scaled)
  y_pred = modelo.predict(X_test_scaled)
  y_prob = modelo.predict_proba(X_test_scaled)[
      :, 1
  ]  # Probabilidad de riesgo (0.0 a 1.0)

  alertas_totales = int(y_pred.sum())
  ataques_cazados = int(((y_pred == 1) & (y_test == 1)).sum())
  falsos_positivos = int(((y_pred == 1) & (y_test == 0)).sum())
  ataques_dejados_fuera = total_ataques_test - ataques_cazados
  pct_dejados_fuera = (ataques_dejados_fuera / total_ataques_test) * 100

  recall = recall_score(y_test, y_pred, zero_division=0)
  precision = precision_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)
  roc_auc = roc_auc_score(y_test, y_prob)

  resultados.append({
      "Algoritmo Supervisado": nombre,
      "Alertas Emitidas": alertas_totales,
      "Ataques Cazados (TP)": ataques_cazados,
      "Dejados Fuera (FN)": ataques_dejados_fuera,
      "% Dejados Fuera": round(pct_dejados_fuera, 2),
      "Falsos Positivos (FP)": falsos_positivos,
      "Recall (%)": round(recall * 100, 2),
      "Precisión (%)": round(precision * 100, 2),
      "F1-Score": round(f1, 4),
      "ROC-AUC": round(roc_auc, 4),
  })

df_res = pd.DataFrame(resultados)
print("\n=== TABLA DE RESULTADOS (MODELOS SUPERVISADOS EN CONJUNTO DE TEST) ===")
print(df_res.to_string(index=False))

# ==============================================================================
# 6. PONDERACIÓN AUTOMÁTICA DE VARIABLES (QUÉ HA APRENDIDO EL RANDOM FOREST)
# ==============================================================================
rf_model = modelos_supervisados["2. Random Forest (200 árboles)"]
importancias = pd.DataFrame({
    "Variable": features,
    "Importancia (%)": np.round(rf_model.feature_importances_ * 100, 2),
}).sort_values(by="Importancia (%)", ascending=False)

print("\n=== RANKING DE IMPORTANCIA DE VARIABLES (RANDOM FOREST) ===")
print(importancias.to_string(index=False))