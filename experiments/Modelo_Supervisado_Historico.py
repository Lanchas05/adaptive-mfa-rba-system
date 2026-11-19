"""
Motor de Autenticación Basada en Riesgo (RBA) y Política Adaptativa Step-Up MFA
================================================================================
- Particionado cronológico estricto: 60% Train / 20% Validación / 20% Test Final
- Codificado de frecuencias aislado (Leak-Free) calculado exclusivamente sobre Train
- Vector fijo de 19 características depuradas (14 base + 5 de afinidad histórica
  por usuario y control de Cold Start / cuenta centinela)
- Comparativa integral: Baseline Heurístico (Reglas IAM) vs. 6 Modelos Supervisados
- Simulación operativa de matriz de decisión adaptativa en tres niveles (Step-Up MFA)
"""

import time
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN GLOBAL DEL EXPERIMENTO
# ==============================================================================
FILE_PATH = "/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv"
N_ROWS = 600000
POS_WEIGHT = 4.0
TH_BAJO = 0.35
TH_ALTO = 0.70

FEATURES_FINALES = [
    # 1. Vector base depurado (14 variables)
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
    # 2. Afinidad histórica del usuario y control de Cold Start / Centinela (5 variables)
    "user_has_history",
    "is_sentinel_user",
    "user_country_ratio_prev",
    "user_asn_ratio_prev",
    "user_ua_ratio_prev",
]


# ==============================================================================
# DEFINICIÓN DEL BASELINE HEURÍSTICO (MOTOR DE REGLAS ESTÁTICAS IAM)
# ==============================================================================
class HeuristicRBABaseline:
  """
  Simula un motor RBA tradicional basado en reglas heurísticas expertas sin ML.
  Asigna pesos aditivos de riesgo [0.0, 1.0] usando únicamente las 19 variables
  disponibles para los modelos supervisados.
  """

  def fit(self, X, y=None):
    return self

  def predict_proba(self, X_raw):
    score = np.zeros(len(X_raw), dtype=float)

    # Regla 1: Password Spraying / Credential Stuffing desde una misma IP
    score += np.where(X_raw["ip_unique_users_prev"] >= 3, 0.35, 0.0)

    # Regla 2: Proveedor de red (ASN) o cruce ASN-País muy infrecuente
    score += np.where(
        (X_raw["ASN_freq"] < 0.005) | (X_raw["asn_x_country_freq"] < 0.002),
        0.25,
        0.0,
    )

    # Regla 3: Usuario con historial previo accediendo desde País y ASN inéditos
    score += np.where(
        (X_raw["user_has_history"] == 1)
        & (X_raw["user_country_ratio_prev"] == 0.0)
        & (X_raw["user_asn_ratio_prev"] == 0.0),
        0.25,
        0.0,
    )

    # Regla 4: Huella de User-Agent automatizada/muy rara o país infrecuente
    score += np.where(
        (X_raw["ua_freq"] < 0.001) | (X_raw["Country_freq"] < 0.01),
        0.20,
        0.0,
    )

    # Regla 5: Sondeo sobre el identificador centinela de cuentas inexistentes
    score += np.where(X_raw["is_sentinel_user"] == 1, 0.15, 0.0)

    score = np.clip(score, 0.0, 1.0)
    return np.column_stack([1.0 - score, score])

  def predict(self, X_raw):
    return (self.predict_proba(X_raw)[:, 1] >= 0.50).astype(int)


# ==============================================================================
# FUNCIONES AUXILIARES DE INGENIERÍA DE CARACTERÍSTICAS SIN DATA LEAKAGE
# ==============================================================================
def count_prev_unique_users(df_in, group_col):
  """Cuenta cuántos User ID distintos han usado una IP antes del evento t."""
  seen = {}
  counts = []
  for entity, user in zip(df_in[group_col], df_in["User ID"]):
    if entity not in seen:
      seen[entity] = set()
    counts.append(len(seen[entity]))
    seen[entity].add(user)
  return counts


def apply_leak_free_frequencies(train, val, test, cols_to_encode):
  """Calcula frecuencias relativas solo en Train y las proyecta en Val y Test."""
  for col, new_name in cols_to_encode.items():
    freq_map = train[col].value_counts(normalize=True)
    train[new_name] = train[col].map(freq_map).fillna(0.0)
    val[new_name] = val[col].map(freq_map).fillna(0.0)
    test[new_name] = test[col].map(freq_map).fillna(0.0)
  return train, val, test


# ==============================================================================
# 1. CARGA Y ORDENACIÓN CRONOLÓGICA DEL DATASET
# ==============================================================================
t0 = time.time()
print(f"[1/6] Cargando {N_ROWS:,} registros desde {FILE_PATH}...")
df = pd.read_csv(FILE_PATH, nrows=N_ROWS)

df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"], errors="coerce")
df = df.sort_values(by="Login Timestamp").reset_index(drop=True)

df["target"] = (
    (df["Is Attack IP"] == True) | (df["Is Account Takeover"] == True)
).astype(int)
df["login_hour"] = df["Login Timestamp"].dt.hour.fillna(12).astype(int)
df["Login_Success_num"] = df["Login Successful"].astype(int)

cat_cols = [
    "Country",
    "ASN",
    "Browser Name and Version",
    "OS Name and Version",
    "Device Type",
    "User Agent String",
]
for col in cat_cols:
  df[col] = df[col].astype(str)

# ==============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS TEMPORAL (ESTRICTO PASADO t-1)
# ==============================================================================
print("[2/6] Construyendo variables históricas y de afinidad por usuario...")

# 2.1 Identificación de User ID centinela (agrupador de cuentas inexistentes)
user_counts = df["User ID"].value_counts()
sentinel_user = user_counts.index[0] if user_counts.iloc[0] > 1000 else None
df["is_sentinel_user"] = (df["User ID"] == sentinel_user).astype(int)

# 2.2 Historial de afinidad por usuario (sin premiar antigüedad bruta)
df["user_prev_logins"] = df.groupby("User ID").cumcount()
df["user_has_history"] = (
    (df["user_prev_logins"] > 0) & (df["is_sentinel_user"] == 0)
).astype(int)

for col, alias in [
    ("Country", "country"),
    ("ASN", "asn"),
    ("User Agent String", "ua"),
]:
  prev_same = df.groupby(["User ID", col]).cumcount()
  df[f"user_{alias}_ratio_prev"] = np.where(
      df["user_has_history"] == 1,
      prev_same / df["user_prev_logins"],
      -1.0,
  )

# 2.3 Desviación horaria respecto al comportamiento histórico del usuario
df["user_hour_mean_prev"] = df.groupby("User ID")["login_hour"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df["user_delta_hour"] = (
    (df["login_hour"] - df["user_hour_mean_prev"]).abs().fillna(0.0)
)

# 2.4 Expansión de cuentas distintas por IP (Password Spraying)
ip_col = "IP Address" if "IP Address" in df.columns else "ASN"
df["ip_unique_users_prev"] = count_prev_unique_users(df, ip_col)

# 2.5 Variables categóricas compuestas para Frequency Encoding
df["asn_x_country"] = df["Country"] + "_" + df["ASN"]
df["hour_x_country"] = df["login_hour"].astype(str) + "_" + df["Country"]
df["hour_x_asn"] = df["login_hour"].astype(str) + "_" + df["ASN"]
df["os_x_browser"] = (
    df["OS Name and Version"] + "_" + df["Browser Name and Version"]
)

# ==============================================================================
# 3. PARTICIONADO CRONOLÓGICO 60/20/20 Y CODIFICADO DE FRECUENCIAS SIN FUGA
# ==============================================================================
print("[3/6] Aplicando split cronológico 60% Train / 20% Val / 20% Test...")
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

cols_frecuencia = {
    "Country": "Country_freq",
    "ASN": "ASN_freq",
    "Browser Name and Version": "Browser Name and Version_freq",
    "OS Name and Version": "OS Name and Version_freq",
    "Device Type": "Device Type_freq",
    "User Agent String": "ua_freq",
    "asn_x_country": "asn_x_country_freq",
    "hour_x_country": "hour_x_country_freq",
    "hour_x_asn": "hour_x_asn_freq",
    "os_x_browser": "os_x_browser_freq",
}

train_df, val_df, test_df = apply_leak_free_frequencies(
    train_df, val_df, test_df, cols_frecuencia
)

# Escalado estándar ajustado exclusivamente en Train
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(train_df[FEATURES_FINALES]),
    columns=FEATURES_FINALES,
    index=train_df.index,
)
y_train = train_df["target"]

X_val = pd.DataFrame(
    scaler.transform(val_df[FEATURES_FINALES]),
    columns=FEATURES_FINALES,
    index=val_df.index,
)
y_val = val_df["target"]

X_test = pd.DataFrame(
    scaler.transform(test_df[FEATURES_FINALES]),
    columns=FEATURES_FINALES,
    index=test_df.index,
)
y_test = test_df["target"]

print(
    f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test:"
    f" {len(X_test):,} | Variables: {len(FEATURES_FINALES)}"
)

# ==============================================================================
# 4. ENTRENAMIENTO Y COMPARATIVA INTEGRAL DE MODELOS (HEURÍSTICO + 6 ML)
# ==============================================================================
print("\n[4/6] Entrenando y evaluando todos los algoritmos del proyecto...")
sample_weights_train = np.where(y_train == 1, POS_WEIGHT, 1.0)

# Formato: (instancia_modelo, usa_sample_weight, usa_datos_crudos_sin_escalar)
modelos = {
    "0. Baseline Heurístico (Reglas IAM)": (HeuristicRBABaseline(), False, True),
    "1. Regresión Logística (Baseline)": (
        LogisticRegression(
            class_weight={0: 1.0, 1: POS_WEIGHT}, max_iter=1000, random_state=42
        ),
        False,
        False,
    ),
    "2. KNN (k=15, pesos distancia)": (
        KNeighborsClassifier(
            n_neighbors=15, weights="distance", leaf_size=50, n_jobs=-1
        ),
        False,
        False,
    ),
    "3. Random Forest (Optimizado)": (
        RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=25,
            class_weight={0: 1.0, 1: POS_WEIGHT},
            random_state=42,
            n_jobs=-1,
        ),
        False,
        False,
    ),
    "4. HistGradientBoosting (Tuned + Peso 4x)": (
        HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=25,
            min_samples_leaf=50,
            l2_regularization=2.0,
            random_state=42,
        ),
        True,
        False,
    ),
    "5. LightGBM (Tuned + Peso 4x)": (
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=25,
            min_child_samples=50,
            reg_lambda=2.0,
            scale_pos_weight=POS_WEIGHT,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        False,
        False,
    ),
    "6. XGBoost (Tuned + Peso 4x)": (
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=50,
            reg_lambda=2.0,
            scale_pos_weight=POS_WEIGHT,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        False,
        False,
    ),
}

resultados_val = []
resultados_test = []
curvas_test = {}
total_ataques_val = int(y_val.sum())
total_ataques_test = int(y_test.sum())

for nombre, (modelo, usa_sample_weight, usa_raw) in modelos.items():
  t_mod = time.time()
  print(f"   -> Ejecutando {nombre}...")

  X_tr_in = train_df[FEATURES_FINALES] if usa_raw else X_train
  X_va_in = val_df[FEATURES_FINALES] if usa_raw else X_val
  X_te_in = test_df[FEATURES_FINALES] if usa_raw else X_test

  if usa_sample_weight:
    modelo.fit(X_tr_in, y_train, sample_weight=sample_weights_train)
  else:
    modelo.fit(X_tr_in, y_train)

  # Evaluación en Validación (20% intermedio)
  prob_va = modelo.predict_proba(X_va_in)[:, 1]
  pred_va = (prob_va >= 0.50).astype(int)
  tp_va = int(((pred_va == 1) & (y_val == 1)).sum())
  fp_va = int(((pred_va == 1) & (y_val == 0)).sum())
  fn_va = total_ataques_val - tp_va

  resultados_val.append({
      "Algoritmo": nombre,
      "ROC-AUC": round(roc_auc_score(y_val, prob_va), 4),
      "PR-AUC": round(average_precision_score(y_val, prob_va), 4),
      "Recall (%)": round(
          recall_score(y_val, pred_va, zero_division=0) * 100, 2
      ),
      "Precisión (%)": round(
          precision_score(y_val, pred_va, zero_division=0) * 100, 2
      ),
      "F1-Score": round(f1_score(y_val, pred_va, zero_division=0), 4),
      "TP": tp_va,
      "FP": fp_va,
      "FN": fn_va,
  })

  # Evaluación en Test Final (20% futuro)
  prob_te = modelo.predict_proba(X_te_in)[:, 1]
  pred_te = (prob_te >= 0.50).astype(int)
  tp_te = int(((pred_te == 1) & (y_test == 1)).sum())
  fp_te = int(((pred_te == 1) & (y_test == 0)).sum())
  fn_te = total_ataques_test - tp_te

  roc_te = roc_auc_score(y_test, prob_te)
  pr_te = average_precision_score(y_test, prob_te)
  fpr_te, tpr_te, _ = roc_curve(y_test, prob_te)
  prec_c_te, rec_c_te, _ = precision_recall_curve(y_test, prob_te)
  curvas_test[nombre] = (
      fpr_te,
      tpr_te,
      prec_c_te,
      rec_c_te,
      roc_te,
      pr_te,
      prob_te,
  )

  resultados_test.append({
      "Algoritmo": nombre,
      "Alertas": int(pred_te.sum()),
      "Cazados (TP)": tp_te,
      "Dejados Fuera (FN)": fn_te,
      "Falsos Positivos (FP)": fp_te,
      "Recall (%)": round(
          recall_score(y_test, pred_te, zero_division=0) * 100, 2
      ),
      "Precisión (%)": round(
          precision_score(y_test, pred_te, zero_division=0) * 100, 2
      ),
      "F1-Score": round(f1_score(y_test, pred_te, zero_division=0), 4),
      "ROC-AUC": round(roc_te, 4),
      "PR-AUC": round(pr_te, 4),
      "Tiempo (s)": round(time.time() - t_mod, 1),
  })

print("\n=== TABLA 1: COMPARATIVA EN VALIDACIÓN CRONOLÓGICA (120.000 FILAS) ===")
print(pd.DataFrame(resultados_val).to_string(index=False))

print("\n=== TABLA 2: EVALUACIÓN FINAL EN TEST CRONOLÓGICO (120.000 FILAS) ===")
print(pd.DataFrame(resultados_test).to_string(index=False))

# ==============================================================================
# 5. IMPORTANCIA DE VARIABLES Y SIMULACIÓN OPERATIVA STEP-UP MFA
# ==============================================================================
print("\n[5/6] Calculando Permutation Importance y Matriz Step-Up MFA en Test...")

# Seleccionamos dinámicamente el mejor modelo según PR-AUC en Validación
df_val_res = pd.DataFrame(resultados_val)
mejor_nombre = df_val_res.sort_values(by="PR-AUC", ascending=False).iloc[0][
    "Algoritmo"
]
print(f"Modelo líder seleccionado según Validación: {mejor_nombre}")

mejor_modelo, _, mejor_usa_raw = modelos[mejor_nombre]
X_test_perm = test_df[FEATURES_FINALES] if mejor_usa_raw else X_test

perm_imp = permutation_importance(
    mejor_modelo,
    X_test_perm,
    y_test,
    scoring="roc_auc",
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

df_perm = pd.DataFrame({
    "Variable": FEATURES_FINALES,
    "Impacto en ROC-AUC": np.round(perm_imp.importances_mean, 5),
    "Desviación": np.round(perm_imp.importances_std, 5),
}).sort_values(by="Impacto en ROC-AUC", ascending=False)

print(f"\n=== RANKING FINAL DE VARIABLES ({len(FEATURES_FINALES)} FEATURES) ===")
print(df_perm.to_string(index=False))

# Simulación de Matriz Adaptativa Step-Up MFA sobre el modelo líder en Test
probs_test_lider = curvas_test[mejor_nombre][6]

banda_baja = probs_test_lider < TH_BAJO
banda_media = (probs_test_lider >= TH_BAJO) & (probs_test_lider < TH_ALTO)
banda_alta = probs_test_lider >= TH_ALTO

total_legitimos_test = int((y_test == 0).sum())

matriz_mfa = pd.DataFrame([
    {
        "Nivel de Riesgo": f"Bajo (P < {TH_BAJO})",
        "Acción IAM": "Acceso Directo (Sin fricción)",
        "Total Accesos": int(banda_baja.sum()),
        "Legítimos (%)": round(
            ((banda_baja & (y_test == 0)).sum() / total_legitimos_test) * 100,
            2,
        ),
        "Ataques (%)": round(
            ((banda_baja & (y_test == 1)).sum() / total_ataques_test) * 100, 2
        ),
    },
    {
        "Nivel de Riesgo": f"Medio ({TH_BAJO} <= P < {TH_ALTO})",
        "Acción IAM": "Desafío Step-Up MFA (OTP/Push)",
        "Total Accesos": int(banda_media.sum()),
        "Legítimos (%)": round(
            ((banda_media & (y_test == 0)).sum() / total_legitimos_test) * 100,
            2,
        ),
        "Ataques (%)": round(
            ((banda_media & (y_test == 1)).sum() / total_ataques_test) * 100, 2
        ),
    },
    {
        "Nivel de Riesgo": f"Crítico (P >= {TH_ALTO})",
        "Acción IAM": "Bloqueo Preventivo",
        "Total Accesos": int(banda_alta.sum()),
        "Legítimos (%)": round(
            ((banda_alta & (y_test == 0)).sum() / total_legitimos_test) * 100,
            2,
        ),
        "Ataques (%)": round(
            ((banda_alta & (y_test == 1)).sum() / total_ataques_test) * 100, 2
        ),
    },
])

print(f"\n=== MATRIZ OPERATIVA STEP-UP MFA EN TEST ({mejor_nombre}) ===")
print(matriz_mfa.to_string(index=False))

# ==============================================================================
# 6. VISUALIZACIÓN COMPARATIVA: CURVAS ROC Y PRECISION-RECALL EN TEST
# ==============================================================================
print("\n[6/6] Generando gráficas comparativas ROC y Precision-Recall...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

for nombre, (fpr, tpr, prec_c, rec_c, roc_auc, pr_auc, _) in curvas_test.items():
  estilo = "--" if "Heurístico" in nombre or "Logística" in nombre else "-"
  grosor = 1.6 if estilo == "--" else 2.1
  ax1.plot(
      fpr,
      tpr,
      linestyle=estilo,
      lw=grosor,
      label=f"{nombre} (AUC={roc_auc:.4f})",
  )
  ax2.plot(
      rec_c,
      prec_c,
      linestyle=estilo,
      lw=grosor,
      label=f"{nombre} (PR-AUC={pr_auc:.4f})",
  )

ax1.plot([0, 1], [0, 1], "k:", lw=1, label="Azar (AUC=0.5000)")
ax1.set_title("Curvas ROC Comparativas (Test Cronológico 120k)")
ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax1.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
ax1.legend(loc="lower right", fontsize=8)
ax1.grid(alpha=0.3)

ax2.set_title("Curvas Precision-Recall (Foco en Clase Minoritaria)")
ax2.set_xlabel("Recall (Fracción de Ataques Cazados)")
ax2.set_ylabel("Precisión (Pureza de las Alertas)")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_roc_pr_final_600k.png", dpi=300)
plt.show()

print(f"\nPipeline finalizado con éxito en {time.time() - t0:.1f} segundos.")