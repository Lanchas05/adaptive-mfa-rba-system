import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')

# 1. Carga de datos
FILE_PATH = '/kaggle/input/datasets/dasgroup/rba-dataset/rba-dataset.csv'
df = pd.read_csv(FILE_PATH, nrows=50000)

# 2. Preprocesamiento global sin variables de usuario
df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
df['login_hour'] = df['Login Timestamp'].dt.hour.fillna(12)
df['RTT_clean'] = df['Round-Trip Time [ms]'].fillna(
    df['Round-Trip Time [ms]'].median()
)
df['Login_Success_num'] = df['Login Successful'].astype(int)

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

features_globales = [
    'login_hour',
    'RTT_clean',
    'Login_Success_num',
    'Country_freq',
    'ASN_freq',
    'Browser Name and Version_freq',
    'OS Name and Version_freq',
    'Device Type_freq',
]

X = df[features_globales]

# Ground Truth de ataques reales
y_true_attack = (
    (df['Is Attack IP'] == True) | (df['Is Account Takeover'] == True)
).astype(int)

# Escalado estándar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Modelos con los hiperparámetros exactos de la captura
modelos_no_supervisados = {
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
    'One-Class SVM (nu=5%)': OneClassSVM(nu=0.05, kernel='rbf', gamma='scale'),
    'DBSCAN (eps=1.2, min=15)': DBSCAN(eps=1.2, min_samples=15, n_jobs=-1),
    'DBSCAN (eps=1.8, min=30)': DBSCAN(eps=1.8, min_samples=30, n_jobs=-1),
}

# 4. Evaluación y formato exacto de salida
resultados = []

for nombre, modelo in modelos_no_supervisados.items():
  preds = modelo.fit_predict(X_scaled)
  anomalias_detectadas = np.where(preds == -1, 1, 0)

  total_anomalias = int(anomalias_detectadas.sum())
  ataques_cazados = int(
      ((anomalias_detectadas == 1) & (y_true_attack == 1)).sum()
  )
  otras_anomalias = int(
      ((anomalias_detectadas == 1) & (y_true_attack == 0)).sum()
  )

  recall = recall_score(y_true_attack, anomalias_detectadas, zero_division=0)
  precision = precision_score(
      y_true_attack, anomalias_detectadas, zero_division=0
  )
  f1 = f1_score(y_true_attack, anomalias_detectadas, zero_division=0)

  resultados.append({
      'Algoritmo y Parámetros': nombre,
      'Anomalías Detectadas': total_anomalias,
      'Ataques Reales Cazados': ataques_cazados,
      'Otras Anomalías (No fraude)': otras_anomalias,
      'Recall (% Ataques pillados)': round(recall * 100, 2),
      'Precisión (%)': round(precision * 100, 2),
      'F1-Score': round(f1, 4),
  })

df_res = pd.DataFrame(resultados)
print('=== TABLA DE RESULTADOS DE LA PRUEBA DE CONCEPTO ===')
print(df_res.to_string(index=False))