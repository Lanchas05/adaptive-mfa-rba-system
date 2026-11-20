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

### 🔹 Sprint 3: Modelado Supervisado, Selección de Variables y Curva ROC (09 Nov – 20 Nov 2026)

* **Objetivo:** Entrenar, auditar y optimizar un clasificador supervisado libre de fuga de datos (*data leakage*) para gobernar una política adaptativa **Step-Up MFA**[cite: 11].
* **Resultado (Review):** Se blindó el *pipeline* mediante particionado cronológico estricto (`60/20/20`) sobre 600.000 registros y se consolidó un vector óptimo de **19 variables** (*Permutation Importance* y ablación)[cite: 8, 9, 11]. **XGBoost** e **HistGradientBoosting** lideraron la comparativa (`ROC-AUC ≈ 0.90`, `PR-AUC = 0.5141 – 0.5207`, `Recall ≈ 79,8%`) frente al *Baseline Heurístico* (`PR-AUC = 0.2007`), neutralizando el **86,40% de los ataques** con un **82,49% de accesos legítimos sin fricción**[cite: 8, 9, 10].
* **Decisión:** Priorizar el **Recall** y la **latencia de inferencia en Test** para elegir el modelo final, evaluar la integración backend (**Odoo** frente a API REST desacoplada) con **MFA agnóstico (TOTP)** descartando APIs de Google, formalizar la gestión tipo **Scrumban/Kanban** y avanzar bajo **GitFlow** hacia el **MVP (`v1.0.0`)** en `main`.
