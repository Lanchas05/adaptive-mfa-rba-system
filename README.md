# adaptive-mfa-rba-system
Sistema de Autenticación Basada en Riesgo (RBA) mediante Machine Learning para la detección de robos de cuenta (ATO) y políticas de autenticación adaptativa (Step-Up MFA).

##  Registro de Sprints (Sprint Reviews)

El desarrollo sigue un marco iterativo e incremental. A continuación se resumen los hitos alcanzados en cada ciclo (el detalle técnico se encuentra en la [Wiki del repositorio](../../wiki)):

### 🔹 Sprint 1: PoC de Detección No Supervisada Global (19 Oct – 30 Oct 2026)
* **Objetivo:** Evaluar la viabilidad de algoritmos no supervisados (*Isolation Forest, LOF, One-Class SVM, DBSCAN*) sobre telemetría global sin contexto histórico.
* **Resultado (Review):** Se procesó una muestra de 50.000 registros aplicando *Frequency Encoding* y escalado estándar. Se demostró empíricamente que los modelos geométricos globales ignoran entre el **88% y el 99% de los ataques reales** (Recall máximo de 11.61% en LOF), debido a que los atacantes camuflan sus accesos bajo parámetros globalmente frecuentes.
* **Decisión:** Descartar la detección global sin estado e incorporar el historial individual por cuenta (`User ID`).

### 🔹 Sprint 2: Contexto Histórico de Usuario y Causalidad Temporal (02 Nov – 06 Nov 2026)
* **Objetivo:** Construir un pipeline de ingeniería de características relativas al historial de cada usuario libre de fuga de datos (*data leakage*) y reevaluar el paradigma no supervisado.
* **Resultado (Review):** Se diseñaron variables dinámicas mediante ventanas expansivas desplazadas (`shift(1).expanding()`) como `user_delta_rtt` y detección de nuevas entidades con conjuntos hash (`set()`). A pesar de enriquecer la señal, los algoritmos no supervisados mantuvieron una tasa inaceptable de falsos negativos (>88%), confirmando la necesidad de guiar el entrenamiento mediante etiquetas reales.
* **Decisión:** Migrar el motor central a **Aprendizaje Supervisado** compensando el desbalanceo de clases (ratio 10:1).

 ### 🔹 Sprint 3: Clasificación Supervisada, Selección de Variables y Curva ROC (09 Nov – 13 Nov 2026) *(En curso)*
* **Objetivo:** Entrenar clasificadores supervisados con balanceo de pesos, depurar variables mediante *Permutation Importance* y calibrar umbrales sobre la curva ROC.
* **Resultado parcial (Review):** 
  * Se incorporaron interacciones cruzadas (`asn_x_country_freq`, `hour_x_country_freq`, `os_x_browser_freq`) que se posicionaron en el Top 5 de impacto predictivo, eliminando variables ruidosas.
  * **HistGradientBoosting** alcanzó un **ROC-AUC de 0.9205** y un **Recall del 88.38%** en el conjunto de test independiente (10.000 filas).
  * El análisis de umbrales sobre la curva ROC demostró que elevar el corte de decisión a la franja `0.65 - 0.70` recorta más de 200 falsos positivos manteniendo un Recall superior al 80%.
