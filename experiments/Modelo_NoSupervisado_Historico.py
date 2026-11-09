import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CARGA DE DATOS Y ORDENACIÓN TEMPORAL POR USUARIO
# ==============================================================================
FILE_PATH = '/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv'
print('Cargando muestra del dataset...')
df = pd.read_csv(FILE_PATH, nrows=50000)

df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
# Ordenación cronológica estricta para garantizar causalidad temporal
df = df.sort_values(by=['User ID', 'Login Timestamp']).reset_index(drop=True)

# Imputaciones básicas
df['login_hour'] = df['Login Timestamp'].dt.hour.fillna(12)
df['RTT_clean'] = df['Round-Trip Time [ms]'].fillna(
    df['Round-Trip Time [ms]'].median()
)
df['Login_Success_num'] = df['Login Successful'].astype(int)

# ==============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS HISTÓRICAS Y CONTEXTUALES (SPRINT 2)
# ==============================================================================
print('Calculando métricas históricas por usuario...')

# 2.1 Frequency Encoding global
cat_cols = [
    'Country',
    'ASN',
    'Browser Name and Version',
    'OS Name and Version',
    'Device Type',
]
for col in cat_cols:
  df[col] = df[col].astype(str)
  freq = df[col].value_counts(normalize=True)
  df[col + '_freq'] = df[col].map(freq)

# 2.2 Desviación de latencia respecto a la media previa histórica del usuario
df['user_rtt_mean_prev'] = df.groupby('User ID')['RTT_clean'].transform(
    lambda x: x.shift(1).expanding().mean()
)
df['user_delta_rtt'] = (
    (df['RTT_clean'] - df['user_rtt_mean_prev']).abs().fillna(0)
)


# 2.3 Banderas de detección de nuevas entidades no vistas en la vida del usuario
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


df['is_new_country_user'] = df.groupby('User ID')['Country'].transform(
    flag_new_entity
)
df['is_new_asn_user'] = df.groupby('User ID')['ASN'].transform(flag_new_entity)

df['device_fingerprint'] = (
    df['Browser Name and Version']
    + '_'
    + df['OS Name and Version']
    + '_'
    + df['Device Type']
)
df['is_new_device_user'] = df.groupby('User ID')[
    'device_fingerprint'
].transform(flag_new_entity)

# 2.4 Fallo de contraseña en el intento inmediatamente anterior
df['prev_failed_login'] = (
    df.groupby('User ID')['Login_Success_num']
    .shift(1)
    .apply(lambda x: 1 if x == 0 else 0)
    .fillna(0)
)

# Vector de 13 variables
features = [
    'login_hour',
    'RTT_clean',
    'Login_Success_num',
    'Country_freq',
    'ASN_freq',
    'Browser Name and Version_freq',
    'OS Name and Version_freq',
    'Device Type_freq',
    'user_delta_rtt',
    'is_new_country_user',
    'is_new_asn_user',
    'is_new_device_user',
    'prev_failed_login',
]

X = df[features]
y_true_attack = (
    (df['Is Attack IP'] == True) | (df['Is Account Takeover'] == True)
).astype(int)
total_ataques = int(y_true_attack.sum())

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 3. EVALUACIÓN NO SUPERVISADA CON CONTEXTO HISTÓRICO
# ==============================================================================
modelos = {
    'Isolation Forest (cont=5%)': IsolationForest(
        contamination=0.05, random_state=42, n_jobs=-1
    ),
    'Isolation Forest (cont=10%)': IsolationForest(
        contamination=0.10, random_state=42, n_jobs=-1
    ),
    'Local Outlier Factor (vecinos=50)': LocalOutlierFactor(
        n_neighbors=50, contamination=0.05, n_jobs=-1
    ),
    'Local Outlier Factor (vecinos=100)': LocalOutlierFactor(
        n_neighbors=100, contamination=0.10, n_jobs=-1
    ),
}

resultados = []
for nombre, modelo in modelos.items():
  preds = modelo.fit_predict(X_scaled)
  anomalias = np.where(preds == -1, 1, 0)

  total_anomalias = int(anomalias.sum())
  cazados = int(((anomalias == 1) & (y_true_attack == 1)).sum())
  otras_anomalias = int(((anomalias == 1) & (y_true_attack == 0)).sum())
  dejados_fuera = total_ataques - cazados
  pct_fuera = (dejados_fuera / total_ataques) * 100

  resultados.append({
      'Algoritmo': nombre,
      'Anomalías Detectadas': total_anomalias,
      'Ataques Cazados': cazados,
      'Dejados Fuera (FN)': dejados_fuera,
      '% Dejados Fuera': round(pct_fuera, 2),
      'Falsos Positivos': otras_anomalias,
      'Recall (%)': round(
          recall_score(y_true_attack, anomalias, zero_division=0) * 100, 2
      ),
      'Precisión (%)': round(
          precision_score(y_true_attack, anomalias, zero_division=0) * 100, 2
      ),
      'F1-Score': round(
          f1_score(y_true_attack, anomalias, zero_division=0), 4
      ),
  })

df_res = pd.DataFrame(resultados)
print(
    '\n=== RESULTADOS: NO SUPERVISADO CON VARIABLES HISTÓRICAS DE USUARIO ==='
)
print(df_res.to_string(index=False))