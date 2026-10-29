import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On simule 1000 mesures de signal radio
np.random.seed(42)
t = np.linspace(0, 100, 1000)

# Signal RSSI normal (puissance du signal, en dBm)
rssi = -70 + 5 * np.sin(t * 0.3) + np.random.normal(0, 1.5, 1000)

# Signal SNR normal (qualité du signal)
snr = rssi + 20 + np.random.normal(0, 0.5, 1000)

# Injection d'anomalies (pannes simulées)
rssi[200:220] = -95   # chute soudaine (panne antenne)
rssi[500:505] = -30   # pic anormal (interférence)
rssi[750:760] = -100  # signal perdu

# Création du DataFrame
df = pd.DataFrame({
    'time': t,
    'RSSI': rssi,
    'SNR': snr,
    'is_anomaly': 0
})
df.loc[200:219, 'is_anomaly'] = 1
df.loc[500:504, 'is_anomaly'] = 1
df.loc[750:759, 'is_anomaly'] = 1 

# Sauvegarde
df.to_csv('signal_data.csv', index=False)
print("Fichier créé :", len(df), "mesures")
print(df.describe())