import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# =========================
# CONFIGURACIÓN
# =========================
CSV_FILE = "eeg_ojos_abiertos_cerrados.csv"   # nombre de tu archivo
FS = 128                                      # frecuencia de muestreo del Insight
CHANNEL = "AF3"                                # prueba también con "T7" o "T8"

WINDOW_SEC = 2.0                              # tamaño de ventana en segundos
STEP_SEC = 0.5                                # desplazamiento entre ventanas

# Bandas EEG
THETA_BAND = (4, 7)
ALPHA_BAND = (8, 12)
BETA_BAND  = (13, 30)



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

signal = df[CHANNEL].values.astype(float)

# Filtrado
signal_filt = bandpass_filter(signal, 1, 40, FS)
signal_filt = signal_filt - np.mean(signal_filt)

# =========================
# CÁLCULO DE BANDAS EN EL TIEMPO
# =========================
window_samples = int(WINDOW_SEC * FS)
step_samples = int(STEP_SEC * FS)

times = []
theta_vals = []
alpha_vals = []
beta_vals = []

for start in range(0, len(signal_filt) - window_samples + 1, step_samples):
    end = start + window_samples
    segment = signal_filt[start:end]

    freqs, psd = welch(segment, fs=FS, nperseg=min(256, len(segment)))

    theta = bandpower_from_psd(freqs, psd, 4, 7)
    alpha = bandpower_from_psd(freqs, psd, 8, 12)
    beta  = bandpower_from_psd(freqs, psd, 13, 30)

    total = bandpower_from_psd(freqs, psd, 1, 40)

    # Potencia relativa
    if total > 0:
        theta = theta / total
        alpha = alpha / total
        beta = beta / total

    center_time = df["t_rel"].iloc[start:end].mean()

    times.append(center_time)
    theta_vals.append(theta)
    alpha_vals.append(alpha)
    beta_vals.append(beta)

# =========================
# GUARDAR RESULTADOS
# =========================
result_df = pd.DataFrame({
    "time": times,
    "theta_rel": theta_vals,
    "alpha_rel": alpha_vals,
    "beta_rel": beta_vals
})

output_csv = f"bandas_tiempo_{CHANNEL}.csv"
result_df.to_csv(output_csv, index=False)
print(f"Resultados guardados en: {output_csv}")

# =========================
# GRÁFICAS
# =========================
plt.figure(figsize=(12, 8))

# Señal filtrada
plt.subplot(2, 1, 1)
plt.plot(df["t_rel"], signal_filt)
plt.title(f"Señal EEG filtrada - {CHANNEL}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)

# Bandas en el tiempo
plt.subplot(2, 1, 2)
plt.plot(times, theta_vals, label="Theta (4-7 Hz)")
plt.plot(times, alpha_vals, label="Alpha (8-12 Hz)")
plt.plot(times, beta_vals, label="Beta (13-30 Hz)")
plt.title(f"Bandas EEG en el tiempo - {CHANNEL}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Potencia relativa")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()