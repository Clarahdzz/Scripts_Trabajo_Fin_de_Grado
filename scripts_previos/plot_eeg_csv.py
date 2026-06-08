import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo CSV
df = pd.read_csv("eeg_ojos_abiertos_cerrados.csv")

# Crear tiempo relativo en segundos
df["t_rel"] = df["time"] - df["time"].iloc[0]

# Canales EEG del Insight
channels = ["AF3", "T7", "Pz", "T8", "AF4"]

plt.figure(figsize=(12,6))

offset = 300   # separación vertical entre señales

for i, ch in enumerate(channels):
    
    signal = df[ch] - df[ch].mean()   # centrar señal
    
    plt.plot(
        df["t_rel"],
        signal + i*offset,
        label=ch
    )

plt.title("Señales EEG - Emotiv Insight")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (offset)")
plt.yticks([i*offset for i in range(len(channels))], channels)
plt.grid(True)

plt.tight_layout()
plt.show()