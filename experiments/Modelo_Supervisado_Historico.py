import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==============================================================================
# 1. CARGA DE MUESTRA DEL RBA DATASET
# ==============================================================================
FILE_PATH = "/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv"

print("Cargando muestra del dataset...")
df = pd.read_csv(FILE_PATH, nrows=50000)

df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"], errors="coerce")
df = df.sort_values(by=["User ID", "Login Timestamp"]).reset_index(drop=True)

df["login_hour"] = df["Login Timestamp"].dt.hour.fillna(12).astype(int)
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
# 2. INGENIERÍA DE CARACTERÍSTICAS POR USUARIO Y CRUCES DE INTERACCIÓN
# ==============================================================================
print("Calculando histórico y nuevas variables contextuales por usuario...")

# 2.1 Desviación de latencia histórica
df["user_rtt_mean_prev"] = df.groupby("User ID")["RTT_clean"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df["user_delta_rtt"] = (
    (df["RTT_clean"] - df["user_rtt_mean_prev"]).abs().fillna(0)
)


# 2.2 Detección de nuevo dispositivo en la historia del usuario
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

# 2.3 Tiempo en segundos desde el último login (detección de fuerza bruta/diccionario)
df["time_since_last_login"] = (
    df.groupby("User ID")["Login Timestamp"]
    .diff()
    .dt.total_seconds()
    .fillna(999999)
)
df["time_since_last_login"] = df["time_since_last_login"].clip(
    lower=0, upper=604800
)

# 2.4 Interacción cruzada País + ASN
df["asn_x_country"] = df["Country"] + "_" + df["ASN"]
df["asn_x_country_freq"] = df["asn_x_country"].map(
    df["asn_x_country"].value_counts(normalize=True)
)

# --- NUEVAS CARACTERÍSTICAS ACORDADAS ---

# 2.5 Interacción cruzada Hora + País
df["hour_x_country"] = df["login_hour"].astype(str) + "_" + df["Country"]
df["hour_x_country_freq"] = df["hour_x_country"].map(
    df["hour_x_country"].value_counts(normalize=True)
)

# 2.6 Interacción cruzada Hora + ASN
df["hour_x_asn"] = df["login_hour"].astype(str) + "_" + df["ASN"]
df["hour_x_asn_freq"] = df["hour_x_asn"].map(
    df["hour_x_asn"].value_counts(normalize=True)
)

# 2.7 Desviación de la hora habitual de conexión del usuario (sin leakage)
df["user_hour_mean_prev"] = df.groupby("User ID")["login_hour"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df["user_delta_hour"] = (
    (df["login_hour"] - df["user_hour_mean_prev"]).abs().fillna(0)
)

# 2.8 Indicador de acceso en horario nocturno (00:00 a 06:00)
df["is_night_login"] = df["login_hour"].apply(
    lambda h: 1 if 0 <= h <= 6 else 0
)

# 2.9 Interacción cruzada Sistema Operativo + Navegador
df["os_x_browser"] = (
    df["OS Name and Version"] + "_" + df["Browser Name and Version"]
)
df["os_x_browser_freq"] = df["os_x_browser"].map(
    df["os_x_browser"].value_counts(normalize=True)
)

# ==============================================================================
# 3. PREPARACIÓN SUPERVISADA: DIVISIÓN TRAIN / TEST (80% / 20%)
# ==============================================================================
# Se han eliminado las 5 variables con impacto nulo/negativo y añadido las 5 nuevas
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
    "is_new_device_user",
    "time_since_last_login",
    "asn_x_country_freq",
    "hour_x_country_freq",
    "hour_x_asn_freq",
    "user_delta_hour",
    "is_night_login",
    "os_x_browser_freq",
]

X = df[features]
y = ((df["Is Attack IP"] == True) | (df["Is Account Takeover"] == True)).astype(
    int
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

total_ataques_test = int(y_test.sum())
print(
    f"Filas Entrenamiento (80%): {len(X_train)} | Filas Test (20%):"
    f" {len(X_test)}"
)
print(
    f"Ataques reales en Test: {total_ataques_test} | Total variables:"
    f" {len(features)}\n"
)

# ==============================================================================
# 4. DEFINICIÓN DE MODELOS (INCLUYENDO KNN Y REGULARIZACIÓN ANTI-OVERFITTING)
# ==============================================================================
modelos_supervisados = {
    "1. Regresión Logística (Baseline)": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "2. KNN (k=15, pesos distancia)": KNeighborsClassifier(
        n_neighbors=15, weights="distance", n_jobs=-1
    ),
    "3. Random Forest (Reg. Profundidad)": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "4. HistGradientBoosting (Estándar)": HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=200, random_state=42
    ),
    "5. HistGradientBoosting (Regularizado)": HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.5,
        random_state=42,
    ),
}

# ==============================================================================
# 5. ENTRENAMIENTO, EVALUACIÓN Y CAPTURA DE CURVAS ROC
# ==============================================================================
resultados = []
curvas_roc = {}

for nombre, modelo in modelos_supervisados.items():
  print(f"Entrenando y evaluando {nombre}...")
  modelo.fit(X_train_scaled, y_train)

  y_pred = modelo.predict(X_test_scaled)
  y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

  # Guardamos datos para el plot de la curva ROC
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  roc_auc = roc_auc_score(y_test, y_prob)
  curvas_roc[nombre] = (fpr, tpr, thresholds, roc_auc, y_prob)

  alertas_totales = int(y_pred.sum())
  ataques_cazados = int(((y_pred == 1) & (y_test == 1)).sum())
  falsos_positivos = int(((y_pred == 1) & (y_test == 0)).sum())
  ataques_dejados_fuera = total_ataques_test - ataques_cazados
  pct_dejados_fuera = (ataques_dejados_fuera / total_ataques_test) * 100

  recall = recall_score(y_test, y_pred, zero_division=0)
  precision = precision_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)

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
print(
    "\n=== TABLA DE RESULTADOS CON 17 VARIABLES DEPURADAS (CORTE POR DEFECTO P"
    " >= 0.50) ==="
)
print(df_res.to_string(index=False))

# ==============================================================================
# 6. FILTRADO DE CARACTERÍSTICAS (PERMUTATION IMPORTANCE EN TEST)
# ==============================================================================
print(
    "\nCalculando Permutation Importance sobre HistGradientBoosting"
    " Regularizado..."
)
hgb_reg = modelos_supervisados["5. HistGradientBoosting (Regularizado)"]
perm_imp = permutation_importance(
    hgb_reg,
    X_test_scaled,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

df_perm = pd.DataFrame({
    "Variable": features,
    "Impacto en ROC-AUC": np.round(perm_imp.importances_mean, 5),
    "Desviación": np.round(perm_imp.importances_std, 5),
}).sort_values(by="Impacto en ROC-AUC", ascending=False)

print(
    "\n=== RANKING REAL DE VARIABLES (PERMUTATION IMPORTANCE EN ROC-AUC) ==="
)
print(df_perm.to_string(index=False))

# ==============================================================================
# 7. ANÁLISIS DE THRESHOLDS SOBRE LA CURVA ROC (REDUCCIÓN DE FALSOS POSITIVOS)
# ==============================================================================
_, _, _, _, probs_hgb = curvas_roc["5. HistGradientBoosting (Regularizado)"]
cortes_prueba = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
analisis_cortes = []

for th in cortes_prueba:
  pred_th = (probs_hgb >= th).astype(int)
  tp = int(((pred_th == 1) & (y_test == 1)).sum())
  fp = int(((pred_th == 1) & (y_test == 0)).sum())
  fn = total_ataques_test - tp

  analisis_cortes.append({
      "Threshold (Probabilidad)": th,
      "Alertas Totales": int(pred_th.sum()),
      "Ataques Cazados (TP)": tp,
      "Falsos Negativos (FN)": fn,
      "Falsos Positivos (FP)": fp,
      "Recall (%)": round((tp / total_ataques_test) * 100, 2),
      "Precisión (%)": round(
          precision_score(y_test, pred_th, zero_division=0) * 100, 2
      ),
  })

df_th = pd.DataFrame(analisis_cortes)
print(
    "\n=== EFECTO DE MOVER EL THRESHOLD EN HISTGRADIENTBOOSTING REGULARIZADO"
    " ==="
)
print(df_th.to_string(index=False))

# ==============================================================================
# 8. GRÁFICA DE LA CURVA ROC COMPARATIVA
# ==============================================================================
plt.figure(figsize=(10, 7))
for nombre, (fpr, tpr, _, roc_auc, _) in curvas_roc.items():
  plt.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {roc_auc:.4f})")

plt.plot(
    [0, 1],
    [0, 1],
    color="navy",
    lw=1.5,
    linestyle="--",
    label="Azar (AUC = 0.5000)",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel(
    "Tasa de Falsos Positivos (FPR - Usuarios legítimos alertados)", fontsize=11
)
plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=11)
plt.title(
    "Curvas ROC Comparativas - Detección de Account Takeover (RBA)", fontsize=13
)
plt.legend(loc="lower right", fontsize=9.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()