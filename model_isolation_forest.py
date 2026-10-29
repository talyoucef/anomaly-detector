import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# 1. Charger les données
df = pd.read_csv('signal_data.csv')
X = df[['RSSI', 'SNR']].values   # les features (colonnes d'entrée)
y_true = df['is_anomaly'].values  # la vérité terrain

# 2. Créer et entraîner le modèle
model = IsolationForest(
    contamination=0.05,  # on estime 5% d'anomalies
    random_state=42
)
model.fit(X)

# 3. Prédire (-1 = anomalie, 1 = normal)
y_pred_raw = model.predict(X)
y_pred = (y_pred_raw == -1).astype(int)  # convertit en 0/1

# 4. Évaluer
print("=== Résultats Isolation Forest ===")
print(classification_report(y_true, y_pred,
      target_names=['Normal', 'Anomalie']))

# 5. Visualiser
plt.figure(figsize=(12, 4))
plt.plot(df['time'], df['RSSI'], color='steelblue',
         linewidth=0.8, label='Signal')
detected = df[y_pred == 1]
plt.scatter(detected['time'], detected['RSSI'],
            color='orange', s=15, label='Détecté par le modèle')
real = df[y_true == 1]
plt.scatter(real['time'], real['RSSI'],
            color='red', s=8, label='Vraie anomalie', marker='x')
plt.legend()
plt.title('Isolation Forest - Détection d\'anomalies')
plt.savefig('isolation_forest_results.png', dpi=150)
plt.show()