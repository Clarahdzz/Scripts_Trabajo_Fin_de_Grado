import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

# =========================================================
# CONFIGURACIÓN
# =========================================================
INPUT_DIR  = "reposo_antiguo"   # carpeta con los CSV antiguos
OUTPUT_DIR = "experimento_3"    # carpeta donde van los convertidos
FS         = 128

CHANNEL_NAMES = ["AF3", "T7", "Pz", "T8", "AF4"]

BANDS_PER_CHANNEL = {
    "AF3": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
    "T7":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "Pz":  [("alpha", 8, 12), ("beta",  13, 30)],
    "T8":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "AF4": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
}

BAND_COLS = []
for ch in CHANNEL_NAMES:
    for band_name, _, _ in BANDS_PER_CHANNEL[ch]:
        BAND_COLS.append(f"{ch}_{band_name}")

CSV_HEADER = ["timestamp"] + CHANNEL_NAMES + BAND_COLS
MS_PER_SAMPLE = 1000 / FS

# =========================================================
# FUNCIONES
# =========================================================
def bandpass_filter(signal, lowcut, highcut, fs=FS, order=4):
    nyquist = 0.5 * fs
    low  = max(1e-4, min(lowcut  / nyquist, 0.9999))
    high = max(1e-4, min(highcut / nyquist, 0.9999))
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def bandpower(signal, fmin, fmax, fs=FS):
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx_band  = (freqs >= fmin) & (freqs <= fmax)
    idx_total = (freqs >= 1)   & (freqs <= 40)
    power_band  = np.trapezoid(psd[idx_band],  freqs[idx_band])
    power_total = np.trapezoid(psd[idx_total], freqs[idx_total])
    if power_total <= 0:
        return 0.0
    return float(power_band / power_total)

def compute_bands(arr):
    features = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = arr[:, ch_idx].astype(float)
        signal = signal - np.mean(signal)
        try:
            signal = bandpass_filter(signal, 1.0, 40.0)
        except Exception:
            pass
        for band_name, fmin, fmax in BANDS_PER_CHANNEL[ch_name]:
            col_name = f"{ch_name}_{band_name}"
            features[col_name] = bandpower(signal, fmin, fmax)
    return features

# =========================================================
# CONVERSIÓN
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Contar cuántos reposo ya existen en OUTPUT_DIR
existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("reposo.") and f.endswith(".csv")]
start_idx = len(existing) + 1

csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
print(f"Encontrados {len(csv_files)} archivos en '{INPUT_DIR}/'")
print(f"Se guardarán a partir de reposo.{start_idx:02d}.csv\n")

converted = 0
for filepath in csv_files:
    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath)

        # Verificar que tiene las columnas necesarias
        for ch in CHANNEL_NAMES:
            if ch not in df.columns:
                print(f"  SKIP {filename} — falta columna {ch}")
                continue

        # Extraer señal cruda (primeras 384 muestras si hay más)
        raw_arr = df[CHANNEL_NAMES].values[:384].astype(float)

        # Si tiene menos de 384 muestras, rellenar con zeros
        if len(raw_arr) < 384:
            pad = np.zeros((384 - len(raw_arr), len(CHANNEL_NAMES)))
            raw_arr = np.vstack([raw_arr, pad])

        # Calcular bandas
        band_features = compute_bands(raw_arr)

        # Guardar en formato experimento_3
        idx      = start_idx + converted
        out_name = f"reposo.{idx:02d}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for i in range(len(raw_arr)):
                timestamp = i * MS_PER_SAMPLE
                raw_vals  = raw_arr[i].tolist()
                band_vals = [band_features[col] for col in BAND_COLS]
                writer.writerow([timestamp] + raw_vals + band_vals)

        print(f"  OK: {filename} → {out_name}")
        converted += 1

    except Exception as e:
        print(f"  ERROR {filename}: {e}")

print(f"\nConvertidos: {converted} archivos")
print(f"Guardados en: {OUTPUT_DIR}/")