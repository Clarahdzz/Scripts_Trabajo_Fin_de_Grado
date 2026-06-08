import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# =========================
# CONFIGURACIÓN
# =========================
CSV_FILE = "eeg_motor_imagery_entrenamiento_v2.csv"
FS = 128

CHANNELS = ["T7", "T8"]

WINDOW_SEC = 2.0
STEP_SEC = 0.5

alpha_BAND = (8, 12)
BETA_BAND = (13, 30)

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

def compute_band_curves(signal, time_vector, fs, window_sec=2.0, step_sec=0.5):
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    times = []
    alpha_vals = []
    beta_vals = []

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = signal[start:end]
        freqs, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))

        alpha = bandpower_from_psd(freqs, psd, alpha_BAND[0], alpha_BAND[1])
        beta = bandpower_from_psd(freqs, psd, BETA_BAND[0], BETA_BAND[1])
        total = bandpower_from_psd(freqs, psd, 1, 40)

        if total > 0:
            alpha /= total
            beta /= total

        times.append(np.mean(time_vector[start:end]))
        alpha_vals.append(alpha)
        beta_vals.append(beta)

    return np.array(times), np.array(alpha_vals), np.array(beta_vals)

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(CSV_FILE)

required_cols = ["time", "label", "trial", "state", "T7", "T8"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Falta la columna requerida: {col}")

df["t_rel"] = df["time"] - df["time"].iloc[0]

# =========================
# FILTRADO
# =========================
for ch in CHANNELS:
    df[f"{ch}_filt"] = bandpass_filter(df[ch].values.astype(float), 1, 40, FS)
    df[f"{ch}_filt"] = df[f"{ch}_filt"] - df[f"{ch}_filt"].mean()

# =========================
# BANDAS EN EL TIEMPO
# =========================
t7_band_t, t7_alpha, t7_beta = compute_band_curves(
    df["T7_filt"].values, df["t_rel"].values, FS, WINDOW_SEC, STEP_SEC
)

t8_band_t, t8_alpha, t8_beta = compute_band_curves(
    df["T8_filt"].values, df["t_rel"].values, FS, WINDOW_SEC, STEP_SEC
)

# =========================
# GRÁFICAS
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# -------- 1) Señales EEG filtradas --------
axes[0].plot(df["t_rel"], df["T7_filt"], label="T7")
axes[0].plot(df["t_rel"], df["T8_filt"], label="T8")
axes[0].set_title("Señales EEG filtradas (T7 y T8)")
axes[0].set_ylabel("Amplitud")
axes[0].grid(True)
axes[0].legend()

# Pintar franjas según etiqueta
for i in range(len(df) - 1):
    x0 = df["t_rel"].iloc[i]
    x1 = df["t_rel"].iloc[i + 1]
    label = df["label"].iloc[i]

    if label == 0:
        axes[0].axvspan(x0, x1, alpha=0.03)
    elif label == 1:
        axes[0].axvspan(x0, x1, alpha=0.08)

# -------- 2) Bandas T7 --------
axes[1].plot(t7_band_t, t7_alpha, label="T7 alpha (8-12 Hz)")
axes[1].plot(t7_band_t, t7_beta, label="T7 Beta (13-30 Hz)")
axes[1].set_title("Potencia relativa de bandas en T7")
axes[1].set_ylabel("Potencia relativa")
axes[1].grid(True)
axes[1].legend()

for i in range(len(df) - 1):
    x0 = df["t_rel"].iloc[i]
    x1 = df["t_rel"].iloc[i + 1]
    label = df["label"].iloc[i]

    if label == 0:
        axes[1].axvspan(x0, x1, alpha=0.03)
    elif label == 1:
        axes[1].axvspan(x0, x1, alpha=0.08)

# -------- 3) Bandas T8 --------
axes[2].plot(t8_band_t, t8_alpha, label="T8 alpha (8-12 Hz)")
axes[2].plot(t8_band_t, t8_beta, label="T8 Beta (13-30 Hz)")
axes[2].set_title("Potencia relativa de bandas en T8")
axes[2].set_xlabel("Tiempo (s)")
axes[2].set_ylabel("Potencia relativa")
axes[2].grid(True)
axes[2].legend()

for i in range(len(df) - 1):
    x0 = df["t_rel"].iloc[i]
    x1 = df["t_rel"].iloc[i + 1]
    label = df["label"].iloc[i]

    if label == 0:
        axes[2].axvspan(x0, x1, alpha=0.03)
    elif label == 1:
        axes[2].axvspan(x0, x1, alpha=0.08)

plt.tight_layout()
plt.show()
