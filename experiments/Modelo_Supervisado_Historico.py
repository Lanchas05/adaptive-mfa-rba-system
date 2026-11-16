import warnings
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CARGA AMPLIADA DEL DATASET (200.000 REGISTROS PARA DENSIDAD HISTÓRICA)
# ==============================================================================
FILE_PATH = "/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv"

print("1. Cargando muestra ampliada (200.000 registros)...")
df = pd.read_csv(FILE_PATH, nrows=200000)

df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"], errors="coerce")

# Ordenación global cronológica inicial para variables de red (IP compartida)
df = df.sort_values(by="Login Timestamp").reset_index(drop=True)

df["login_hour"] = df["Login Timestamp"].dt.hour.fillna(12).astype(int)
df["Login_Success_num"] = df["Login Successful"].astype(int)

# ==============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS DEPURADA (14 VARIABLES ÓPTIMAS)
# ==============================================================================
print(
    "2. Generando características seleccionadas de red, usuario e"
    " interacciones..."
)

# 2.1 Detección de Password Spraying / Credential Stuffing (ip_unique_users_prev)
ip_col = "IP Address" if "IP Address" in df.columns else "ASN"


def count_prev_unique_users(df_in, group_col):
  seen_users_per_ip = {}
  counts = []
  for ip, user in zip(df_in[group_col], df_in["User ID"]):
    if ip not in seen_users_per_ip:
      seen_users_per_ip[ip] = set()
    counts.append(len(seen_users_per_ip[ip]))
    seen_users_per_ip[ip].add(user)
  return counts


df["ip_unique_users_prev"] = count_prev_unique_users(df, ip_col)

# Reordenamos por Usuario y Timestamp para calcular el historial individual de cada cuenta
df = df.sort_values(by=["User ID", "Login Timestamp"]).reset_index(drop=True)

# 2.2 Frequency Encoding de variables categóricas base + User-Agent completo (ua_freq)
cat_cols = [
    "Country",
    "ASN",
    "Browser Name and Version",
    "OS Name and Version",
    "Device Type",
]
for col in cat_cols:
  df[col] = df[col].astype(str)
  df[col + "_freq"] = df[col].map(df[col].value_counts(normalize=True))

if "User Agent String" in df.columns:
  df["User Agent String"] = df["User Agent String"].astype(str)
  df["ua_freq"] = df["User Agent String"].map(
      df["User Agent String"].value_counts(normalize=True)
  )
else:
  df["ua_freq"] = (
      df["Browser Name and Version_freq"] * df["OS Name and Version_freq"]
  )

# 2.3 Desviación horaria del usuario (sin data leakage)
df["user_hour_mean_prev"] = df.groupby("User ID")["login_hour"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df["user_delta_hour"] = (
    (df["login_hour"] - df["user_hour_mean_prev"]).abs().fillna(0)
)

# 2.4 Cruces de interacción de alto impacto
df["asn_x_country"] = df["Country"] + "_" + df["ASN"]
df["asn_x_country_freq"] = df["asn_x_country"].map(
    df["asn_x_country"].value_counts(normalize=True)
)

df["hour_x_country"] = df["login_hour"].astype(str) + "_" + df["Country"]
df["hour_x_country_freq"] = df["hour_x_country"].map(
    df["hour_x_country"].value_counts(normalize=True)
)

df["hour_x_asn"] = df["login_hour"].astype(str) + "_" + df["ASN"]
df["hour_x_asn_freq"] = df["hour_x_asn"].map(
    df["hour_x_asn"].value_counts(normalize=True)
)

df["os_x_browser"] = (
    df["OS Name and Version"] + "_" + df["Browser Name and Version"]
)
df["os_x_browser_freq"] = df["os_x_browser"].map(
    df["os_x_browser"].value_counts(normalize=True)
)

# ==============================================================================
# 3. SELECCIÓN FINAL (14 VARIABLES) Y PARTICIONADO ESTRATIFICADO (80% / 20%)
# ==============================================================================
features = [
    "login_hour",
    "Login_Success_num",
    "Country_freq",
    "ASN_freq",
    "Browser Name and Version_freq",
    "OS Name and Version_freq",
    "Device Type_freq",
    "ua_freq",
    "ip_unique_users_prev",
    "user_delta_hour",
    "asn_x_country_freq",
    "hour_x_country_freq",
    "hour_x_asn_freq",
    "os_x_browser_freq",
]

X = df[features]
y = ((df["Is Attack IP"] == True) | (df["Is Account Takeover"] == True)).astype(
    int
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Conservamos los nombres de las columnas tras el escalado para evitar UserWarnings
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=features, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=features, index=X_test.index
)

total_ataques_test = int(y_test.sum())
print(
    f"Filas Entrenamiento: {len(X_train)} | Filas Test: {len(X_test)} | Ataques"
    f" en Test: {total_ataques_test} | Variables: {len(features)}"
)

# ==============================================================================
# 4. CALIBRACIÓN DE PESOS MODERADOS (PENALIZACIÓN 4x)
# ==============================================================================
POS_WEIGHT = 4.0
sample_weights_train = np.where(y_train == 1, POS_WEIGHT, 1.0)

modelos = {
    "1. Regresión Logística (Baseline)": (
        LogisticRegression(
            class_weight={0: 1.0, 1: POS_WEIGHT}, max_iter=1000, random_state=42
        ),
        False,
    ),
    "2. Random Forest (Optimizado)": (
        RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=25,
            class_weight={0: 1.0, 1: POS_WEIGHT},
            random_state=42,
            n_jobs=-1,
        ),
        False,
    ),
    "3. HistGradientBoosting (Tuned + Peso 4x)": (
        HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            max_leaf_nodes=25,
            min_samples_leaf=40,
            l2_regularization=2.0,
            random_state=42,
        ),
        True,
    ),
    "4. LightGBM (Tuned + Peso 4x)": (
        LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=25,
            min_child_samples=40,
            reg_lambda=2.0,
            scale_pos_weight=POS_WEIGHT,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        False,
    ),
    "5. XGBoost (Tuned + Peso 4x)": (
        XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=40,
            reg_lambda=2.0,
            scale_pos_weight=POS_WEIGHT,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        False,
    ),
}

# ==============================================================================
# 5. ENTRENAMIENTO Y EVALUACIÓN (ROC-AUC + PR-AUC)
# ==============================================================================
resultados = []
curvas = {}

for nombre, (modelo, usa_sample_weight) in modelos.items():
  print(f"Entrenando {nombre}...")
  if usa_sample_weight:
    modelo.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
  else:
    modelo.fit(X_train_scaled, y_train)

  y_pred = modelo.predict(X_test_scaled)
  y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

  fpr, tpr, _ = roc_curve(y_test, y_prob)
  prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

  roc_auc = roc_auc_score(y_test, y_prob)
  pr_auc = average_precision_score(y_test, y_prob)
  curvas[nombre] = (fpr, tpr, prec_curve, rec_curve, roc_auc, pr_auc, y_prob)

  tp = int(((y_pred == 1) & (y_test == 1)).sum())
  fp = int(((y_pred == 1) & (y_test == 0)).sum())
  fn = total_ataques_test - tp

  resultados.append({
      "Algoritmo": nombre,
      "Alertas": int(y_pred.sum()),
      "Cazados (TP)": tp,
      "Dejados Fuera (FN)": fn,
      "% Dejados Fuera": round((fn / total_ataques_test) * 100, 2),
      "Falsos Positivos (FP)": fp,
      "Recall (%)": round(
          recall_score(y_test, y_pred, zero_division=0) * 100, 2
      ),
      "Precisión (%)": round(
          precision_score(y_test, y_pred, zero_division=0) * 100, 2
      ),
      "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
      "ROC-AUC": round(roc_auc, 4),
      "PR-AUC": round(pr_auc, 4),
  })

df_res = pd.DataFrame(resultados)
print(
    "\n=== TABLA COMPARATIVA DE EFICIENCIA (14 VARIABLES DEPURADAS | CORTE 0.50)"
    " ==="
)
print(df_res.to_string(index=False))

# ==============================================================================
# 6. RANKING DE IMPORTANCIA DE VARIABLES (PERMUTATION IMPORTANCE)
# ==============================================================================
print("\nCalculando Permutation Importance sobre LightGBM...")
mejor_lgbm = modelos["4. LightGBM (Tuned + Peso 4x)"][0]
perm_imp = permutation_importance(
    mejor_lgbm,
    X_test_scaled,
    y_test,
    scoring="roc_auc",
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

df_perm = pd.DataFrame({
    "Variable": features,
    "Impacto en ROC-AUC": np.round(perm_imp.importances_mean, 5),
    "Desviación": np.round(perm_imp.importances_std, 5),
}).sort_values(by="Impacto en ROC-AUC", ascending=False)

print("\n=== RANKING FINAL DE VARIABLES (14 FEATURES) ===")
print(df_perm.to_string(index=False))

# ==============================================================================
# 7. BARRIDO DE THRESHOLDS EN EL MODELO LÍDER (LIGHTGBM)
# ==============================================================================
_, _, _, _, _, _, probs_opt = curvas["4. LightGBM (Tuned + Peso 4x)"]
cortes = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
tabla_th = []

for th in cortes:
  p_th = (probs_opt >= th).astype(int)
  tp = int(((p_th == 1) & (y_test == 1)).sum())
  fp = int(((p_th == 1) & (y_test == 0)).sum())
  fn = total_ataques_test - tp
  tabla_th.append({
      "Threshold": th,
      "Alertas": int(p_th.sum()),
      "Cazados (TP)": tp,
      "Dejados Fuera (FN)": fn,
      "Falsos Positivos (FP)": fp,
      "Recall (%)": round((tp / total_ataques_test) * 100, 2),
      "Precisión (%)": round(
          precision_score(y_test, p_th, zero_division=0) * 100, 2
      ),
  })

print("\n=== ANÁLISIS DE THRESHOLDS (LIGHTGBM TUNED) ===")
print(pd.DataFrame(tabla_th).to_string(index=False))

# ==============================================================================
# 8. DOBLE GRÁFICA: CURVA ROC Y CURVA PRECISION-RECALL
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for nombre, (fpr, tpr, prec_c, rec_c, roc_auc, pr_auc, _) in curvas.items():
  ax1.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC={roc_auc:.4f})")
  ax2.plot(rec_c, prec_c, lw=2, label=f"{nombre} (PR-AUC={pr_auc:.4f})")

ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set_title("Curvas ROC Comparativas")
ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax1.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
ax1.legend(loc="lower right", fontsize=8.5)
ax1.grid(alpha=0.3)

ax2.set_title("Curvas Precision-Recall (Foco en Clase Minoritaria)")
ax2.set_xlabel("Recall (Fracción de Ataques Cazados)")
ax2.set_ylabel("Precisión (Pureza de las Alertas)")
ax2.legend(loc="upper right", fontsize=8.5)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()