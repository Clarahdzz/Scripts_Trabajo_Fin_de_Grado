import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# =========================
# CONFIGURACIÓN
# =========================

CSV_FILE = "eeg_ojos_abiertos_cerrados.csv"
FS = 128  # frecuencia de muestreo del Insight
CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"]

BANDS = {
    "Theta": (4, 7),
    "Alpha": (8, 12),
    "Beta": (13, 30),
}

# =========================
# FUNCIONES
# =========================
def bandpass_filter(data, lowcut=1, highcut=40, fs=128, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def bandpower_from_psd(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapezoid(psd[idx], freqs[idx])

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(CSV_FILE)
df["t_rel"] = df["time"] - df["time"].iloc[0]

# =========================
# CÁLCULO DE POTENCIA POR BANDAS
# =========================
results = []

for ch in CHANNELS:
    signal = df[ch].values.astype(float)

    # Filtrado 1-40 Hz
    filtered = bandpass_filter(signal, lowcut=1, highcut=40, fs=FS)

    # Quitar media
    filtered = filtered - np.mean(filtered)

    # PSD con Welch
    freqs, psd = welch(filtered, fs=FS, nperseg=min(256, len(filtered)))

    # Potencia por bandas
    row = {"Canal": ch}
    total_power = bandpower_from_psd(freqs, psd, 1, 40)

    for band_name, (fmin, fmax) in BANDS.items():
        bp = bandpower_from_psd(freqs, psd, fmin, fmax)
        row[band_name] = bp
        row[f"{band_name}_rel"] = bp / total_power if total_power > 0 else 0.0

    results.append(row)

results_df = pd.DataFrame(results)

print("\nPotencia absoluta por bandas:")
print(results_df[["Canal", "Theta", "Alpha", "Beta"]])

print("\nPotencia relativa por bandas:")
print(results_df[["Canal", "Theta_rel", "Alpha_rel", "Beta_rel"]])

# Guardar resultados
results_df.to_csv("bandas_resultados_abierto_cerrado.csv", index=False)
print("\nResultados guardados en bandas_resultados_abierto_cerrado.csv")

# =========================
# GRÁFICA DE POTENCIA RELATIVA
# =========================
x = np.arange(len(CHANNELS))
width = 0.25

theta_vals = results_df["Theta_rel"].values
alpha_vals = results_df["Alpha_rel"].values
beta_vals = results_df["Beta_rel"].values

plt.figure(figsize=(10, 5))
plt.bar(x - width, theta_vals, width, label="Theta (4-7 Hz)")
plt.bar(x, alpha_vals, width, label="Alpha (8-12 Hz)")
plt.bar(x + width, beta_vals, width, label="Beta (13-30 Hz)")

plt.xticks(x, CHANNELS)
plt.ylabel("Potencia relativa")
plt.title("Potencia relativa por bandas EEG")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()