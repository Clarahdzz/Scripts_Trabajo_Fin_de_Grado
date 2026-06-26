import json
import ssl
from collections import deque, Counter

import websocket
import numpy as np
import pandas as pd
import joblib
from scipy.signal import welch, butter, filtfilt, find_peaks


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

CLIENT_ID     = # tu client id
CLIENT_SECRET = # tu client secret
CORTEX_URL    = "wss://localhost:6868"

MODEL_PATH = # la caprteta donde tengas las muestras, en este caso están en : ...\experimento_3\modelo_svm_experimento3.pkl"

FS             = 128
WINDOW_SECONDS = 3
WINDOW_SAMPLES = FS * WINDOW_SECONDS  # 384 muestras

PREDICT_EVERY_SAMPLES = 64  # predicción cada ~0.5s
N_LAST_PREDICTIONS    = 5   # voto mayoritario
MIN_CONFIDENCE        = 0.65

# Bandas
THETA_BAND = (4, 7)
ALPHA_BAND = (8, 12)
MU_BAND    = (8, 12)
BETA_BAND  = (13, 30)

# Canales necesarios
CHANNEL_NAMES   = ["AF3", "T7", "Pz", "T8", "AF4"]
CHANNEL_INDICES = {
    "AF3": 2, "T7": 3, "Pz": 4, "T8": 5, "AF4": 6
}

BAND_COLS = [
    "AF3_theta", "AF3_alpha", "AF3_beta",
    "T7_theta",  "T7_mu",     "T7_beta",
    "Pz_alpha",  "Pz_beta",
    "T8_theta",  "T8_mu",     "T8_beta",
    "AF4_theta", "AF4_alpha", "AF4_beta",
]

BANDS_PER_CHANNEL = {
    "AF3": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
    "T7":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "Pz":  [("alpha", 8, 12), ("beta",  13, 30)],
    "T8":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "AF4": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
}


# ============================================================
# 2. CARGAR MODELO
# ============================================================

print("\nCargando modelo...")
model_package = joblib.load(MODEL_PATH)

model         = model_package["model"]
label_encoder = model_package["label_encoder"]
feature_cols  = model_package["feature_cols"]

print("Modelo cargado correctamente.")
print(f"Features esperadas: {len(feature_cols)}")


# ============================================================
# 3. FUNCIONES AUXILIARES CORTEX
# ============================================================

request_id = 1

def send_request(ws, method, params=None):
    global request_id
    message = {
        "jsonrpc": "2.0",
        "method":  method,
        "params":  params if params is not None else {},
        "id":      request_id
    }
    ws.send(json.dumps(message))
    current_id = request_id
    request_id += 1
    while True:
        response = json.loads(ws.recv())
        if response.get("id") == current_id:
            if "error" in response:
                raise RuntimeError(f"Error en {method}: {response['error']}")
            return response.get("result")

def authorize(ws):
    print("\nAutorizando aplicación...")
    result = send_request(ws, "authorize", {
        "clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET
    })
    token = result["cortexToken"]
    print("Autorizado. Token recibido.")
    return token

def query_headsets(ws):
    print("\nBuscando headsets...")
    result = send_request(ws, "queryHeadsets")
    if not result:
        raise RuntimeError("No se ha encontrado ningún headset.")
    headset_id = result[0]["id"]
    print("Headset encontrado:", headset_id)
    return headset_id

def create_session(ws, cortex_token, headset_id):
    print("\nCreando sesión...")
    result = send_request(ws, "createSession", {
        "cortexToken": cortex_token,
        "headset":     headset_id,
        "status":      "active"
    })
    session_id = result["id"]
    print("Sesión creada:", session_id)
    return session_id

def subscribe_eeg(ws, cortex_token, session_id):
    print("\nSuscribiendo a stream EEG...")
    result = send_request(ws, "subscribe", {
        "cortexToken": cortex_token,
        "session":     session_id,
        "streams":     ["eeg"]
    })
    print("Suscripción EEG realizada.")
    return result


# ============================================================
# 4. EXTRACCIÓN DE FEATURES (igual que en entrenamiento)
# ============================================================

def bandpass_filter(signal, lowcut, highcut, fs=FS, order=4):
    nyquist = 0.5 * fs
    low  = max(1e-4, min(lowcut  / nyquist, 0.9999))
    high = max(1e-4, min(highcut / nyquist, 0.9999))
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def band_power(signal, fs, fmin, fmax):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    try:
        signal = bandpass_filter(signal, 1.0, 40.0, fs)
    except Exception:
        pass
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx_band  = (freqs >= fmin) & (freqs <= fmax)
    idx_total = (freqs >= 1)   & (freqs <= 40)
    power_band  = np.trapezoid(psd[idx_band],  freqs[idx_band])
    power_total = np.trapezoid(psd[idx_total], freqs[idx_total])
    if power_total <= 0:
        return 0.0
    return float(power_band / power_total)

def extract_features_from_window(buffers):
    """
    buffers: dict {canal: list de muestras}
    Extrae exactamente las mismas features que en el entrenamiento.
    """
    eps = 1e-10
    features = {}

    # --- Features espectrales ---
    band_values = {}
    for ch in CHANNEL_NAMES:
        signal = np.array(buffers[ch], dtype=float)
        for band_name, fmin, fmax in BANDS_PER_CHANNEL[ch]:
            col = f"{ch}_{band_name}"
            val = band_power(signal, FS, fmin, fmax)
            band_values[col] = val
            features[f"{col}_mean"] = val
            features[f"{col}_std"]  = 0.0  # ventana única → std = 0

    for band in ["theta", "alpha", "beta"]:
        af3_col = f"AF3_{band}"
        af4_col = f"AF4_{band}"
        if af3_col in band_values and af4_col in band_values:
            features[f"AF3_AF4_{band}_diff"]  = band_values[af3_col] - band_values[af4_col]
            features[f"AF3_AF4_{band}_ratio"] = band_values[af3_col] / (band_values[af4_col] + eps)

    for band in ["theta", "mu", "beta"]:
        t7_col = f"T7_{band}"
        t8_col = f"T8_{band}"
        if t7_col in band_values and t8_col in band_values:
            features[f"T7_T8_{band}_diff"] = band_values[t7_col] - band_values[t8_col]

    if "AF3_alpha" in band_values and "AF4_alpha" in band_values:
        af3_a = band_values["AF3_alpha"]
        af4_a = band_values["AF4_alpha"]
        features["frontal_alpha_asymmetry"] = (af4_a - af3_a) / (af4_a + af3_a + eps)

    if "AF3_theta" in band_values and "AF3_beta" in band_values:
        features["AF3_theta_beta_ratio"] = band_values["AF3_theta"] / (band_values["AF3_beta"] + eps)
    if "AF4_theta" in band_values and "AF4_beta" in band_values:
        features["AF4_theta_beta_ratio"] = band_values["AF4_theta"] / (band_values["AF4_beta"] + eps)

    # --- Features de amplitud cruda AF3 y AF4 ---
    for ch in ["AF3", "AF4"]:
        signal = np.array(buffers[ch], dtype=float)
        signal = signal - np.mean(signal)

        features[f"{ch}_raw_max"]      = np.max(signal)
        features[f"{ch}_raw_min"]      = np.min(signal)
        features[f"{ch}_raw_range"]    = np.max(signal) - np.min(signal)
        features[f"{ch}_raw_std"]      = np.std(signal)
        features[f"{ch}_raw_var"]      = np.var(signal)
        features[f"{ch}_raw_abs_mean"] = np.mean(np.abs(signal))

        threshold = np.percentile(np.abs(signal), 75)
        peaks, _ = find_peaks(np.abs(signal), height=threshold, distance=10)
        features[f"{ch}_n_peaks"]       = len(peaks)
        features[f"{ch}_peak_mean_amp"] = np.mean(np.abs(signal[peaks])) if len(peaks) > 0 else 0.0

    X = pd.DataFrame([features])
    X = X[feature_cols]
    return X


# ============================================================
# 5. LOOP DE TIEMPO REAL
# ============================================================

def main():
    print("\nConectando con Cortex...")
    ws = websocket.create_connection(CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE})

    cortex_token = authorize(ws)
    headset_id   = query_headsets(ws)
    session_id   = create_session(ws, cortex_token, headset_id)
    subscribe_eeg(ws, cortex_token, session_id)

    # Buffers para los 5 canales
    buffers = {ch: deque(maxlen=WINDOW_SAMPLES) for ch in CHANNEL_NAMES}

    last_predictions = deque(maxlen=N_LAST_PREDICTIONS)
    eeg_labels   = None
    sample_counter = 0

    print("\n================================================")
    print("TIEMPO REAL INICIADO — EXPERIMENTO 3")
    print("Piensa en la PELOTA BOTANDO, PESTAÑEA o RELÁJATE.")
    print("Pulsa Ctrl+C para parar.")
    print("================================================\n")

    try:
        while True:
            message = json.loads(ws.recv())

            if "eeg" in message and "cols" in message:
                eeg_labels = message["cols"]
                continue

            if "eeg" not in message:
                continue

            eeg_data = message["eeg"]

            if eeg_labels is None:
                eeg_labels = [
                    "COUNTER", "INTERPOLATED", "AF3", "T7", "Pz", "T8", "AF4",
                    "RAW_CQ", "MARKER_HARDWARE"
                ]

            data_dict = dict(zip(eeg_labels, eeg_data))

            # Alimentar buffers
            for ch in CHANNEL_NAMES:
                if ch in data_dict:
                    buffers[ch].append(float(data_dict[ch]))

            sample_counter += 1

            # Esperar a llenar la ventana
            if len(buffers["AF3"]) < WINDOW_SAMPLES:
                continue

            # Predecir cada PREDICT_EVERY_SAMPLES muestras
            if sample_counter % PREDICT_EVERY_SAMPLES != 0:
                continue

            X_new = extract_features_from_window({ch: list(buffers[ch]) for ch in CHANNEL_NAMES})

            pred_encoded  = model.predict(X_new)[0]
            pred_label    = label_encoder.inverse_transform([pred_encoded])[0]
            last_predictions.append(pred_label)
            final_prediction = Counter(last_predictions).most_common(1)[0][0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_new)[0]
                prob_pairs = {clase: prob for clase, prob in zip(label_encoder.classes_, probs)}
                max_confidence = max(prob_pairs.values())
                prob_text = " | " + " ".join([f"{k}: {v:.2f}" for k, v in prob_pairs.items()])

                if max_confidence >= MIN_CONFIDENCE:
                    print(f"Predicción instantánea: {pred_label} → Decisión final: {final_prediction}{prob_text}")
            else:
                probs = None
                max_confidence = 0.0
                prob_text = ""
                print(f"Predicción instantánea: {pred_label} → Decisión final: {final_prediction}")

    except KeyboardInterrupt:
        print("\nParando tiempo real...")

    finally:
        try:
            ws.close()
        except Exception:
            pass
        print("Conexión cerrada.")


if __name__ == "__main__":
    main()
