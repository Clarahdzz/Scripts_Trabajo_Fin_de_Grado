import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Cargar datos
df = pd.read_csv("eeg_ojos_abiertos_cerrados.csv")

# Tiempo relativo
df["t_rel"] = df["time"] - df["time"].iloc[0]

# Parámetros
fs = 128  # frecuencia de muestreo
lowcut = 1
highcut = 40

# Filtro Butterworth
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Canales
channels = ["AF3", "T7", "Pz", "T8", "AF4"]

plt.figure(figsize=(12,6))
offset = 300

for i, ch in enumerate(channels):
    raw = df[ch]
    filtered = bandpass_filter(raw, lowcut, highcut, fs)
    
    # centrar
    filtered = filtered - np.mean(filtered)

    plt.plot(df["t_rel"], filtered + i*offset, label=ch)

plt.title("EEG filtrado (1–40 Hz)")
plt.xlabel("Tiempo (s)")
plt.yticks([i*offset for i in range(len(channels))], channels)
plt.grid(True)

plt.tight_layout()
plt.show()