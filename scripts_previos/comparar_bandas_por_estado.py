import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

CSV_FILE = "eeg_motor_imagery_entrenamiento_v2.csv"
FS = 128

CHANNELS = ["T7", "T8"]

MU_BAND = (8, 12)
BETA_BAND = (13, 30)

def bandpower(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapezoid(psd[idx], freqs[idx])

# =========================
# CARGA
# =========================
df = pd.read_csv(CSV_FILE)

# =========================
# SEPARAR POR ESTADO
# =========================
df_rest = df[df["label"] == 0]
df_action = df[df["label"] == 1]

results = []

for ch in CHANNELS:

    # reposo
    signal_rest = df_rest[ch].values - np.mean(df_rest[ch].values)
    mu_rest = bandpower(signal_rest, FS, *MU_BAND)
    beta_rest = bandpower(signal_rest, FS, *BETA_BAND)

    # motor imagery
    signal_action = df_action[ch].values - np.mean(df_action[ch].values)
    mu_action = bandpower(signal_action, FS, *MU_BAND)
    beta_action = bandpower(signal_action, FS, *BETA_BAND)

    results.append({
        "canal": ch,
        "mu_rest": mu_rest,
        "mu_action": mu_action,
        "beta_rest": beta_rest,
        "beta_action": beta_action
    })

results_df = pd.DataFrame(results)

# =========================
# GRAFICAR
# =========================
x = np.arange(len(CHANNELS))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width/2, results_df["mu_rest"], width, label="Mu reposo")
ax.bar(x + width/2, results_df["mu_action"], width, label="Mu motor imagery")

ax.set_xticks(x)
ax.set_xticklabels(CHANNELS)
ax.set_ylabel("Potencia")
ax.set_title("Comparación banda MU (T7 vs T8)")
ax.legend()
ax.grid()

plt.show()


fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width/2, results_df["beta_rest"], width, label="Beta reposo")
ax.bar(x + width/2, results_df["beta_action"], width, label="Beta motor imagery")

ax.set_xticks(x)
ax.set_xticklabels(CHANNELS)
ax.set_ylabel("Potencia")
ax.set_title("Comparación banda BETA (T7 vs T8)")
ax.legend()
ax.grid()

plt.show()