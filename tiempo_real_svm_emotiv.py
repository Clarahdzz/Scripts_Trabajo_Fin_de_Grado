import json
import ssl
import time
from collections import deque, Counter

import websocket
import numpy as np
import pandas as pd
import joblib
from scipy.signal import welch, butter, filtfilt


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

CLIENT_ID     = # tu client id
CLIENT_SECRET = # tu client secret
CORTEX_URL = "wss://localhost:6868"

MODEL_PATH = r"C:\Users\Emilio\Desktop\4_TELECO\TFG_EEG\valores_com_emotiv_128hz\modelo_svm_rbf_t7t8.pkl"

FS = 128  

WINDOW_SECONDS = 3
WINDOW_SAMPLES = FS * WINDOW_SECONDS

# Cada cuántas muestras se hace una predicción.
# 64 muestras = cada 0.5 s aproximadamente
PREDICT_EVERY_SAMPLES = 64

# Voto mayoritario para suavizar la salida
N_LAST_PREDICTIONS = 5

# Bandas usadas
THETA_BAND = (4, 7)
MU_BAND = (8, 12)
BETA_BAND = (13, 30)


# ============================================================
# 2. CARGAR MODELO
# ============================================================

print("\nCargando modelo...")
model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
label_encoder = model_package["label_encoder"]
feature_cols = model_package["feature_cols"]

print("Modelo cargado correctamente.")
print("Features esperadas:")
for f in feature_cols:
    print("-", f)


# ============================================================
# 3. FUNCIONES AUXILIARES CORTEX
# ============================================================

request_id = 1


def send_request(ws, method, params=None):
    global request_id

    message = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params if params is not None else {},
        "id": request_id
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
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET
    })

    token = result["cortexToken"]
    print("Autorizado. Token recibido.")
    return token


def query_headsets(ws):
    print("\nBuscando headsets...")

    result = send_request(ws, "queryHeadsets")
    if not result:
        raise RuntimeError("No se ha encontrado ningún headset. Revisa conexión y Emotiv Launcher.")

    headset_id = result[0]["id"]
    print("Headset encontrado:", headset_id)
    return headset_id


def create_session(ws, cortex_token, headset_id):
    print("\nCreando sesión...")

    result = send_request(ws, "createSession", {
        "cortexToken": cortex_token,
        "headset": headset_id,
        "status": "active"
    })

    session_id = result["id"]
    print("Sesión creada:", session_id)
    return session_id


def subscribe_eeg(ws, cortex_token, session_id):
    print("\nSuscribiendo a stream EEG...")

    result = send_request(ws, "subscribe", {
        "cortexToken": cortex_token,
        "session": session_id,
        "streams": ["eeg"]
    })

    print("Suscripción EEG realizada.")
    print(result)
    return result


# ============================================================
# 4. CÁLCULO DE BANDAS
# ============================================================  

def bandpass_filter(signal, lowcut, highcut, fs=FS, order=4):
    nyquist = 0.5 * fs
    low = max(1e-4, min(lowcut / nyquist, 0.9999))
    high = max(1e-4, min(highcut / nyquist, 0.9999))
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def band_power(signal, fs, band):
    """Potencia RELATIVA de una banda, igual que en la captura."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)          # quitar DC

    # filtro general 1-40 Hz antes de calcular bandas
    try:
        signal = bandpass_filter(signal, 1.0, 40.0, fs)
    except Exception:
        pass

    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    idx_band = (freqs >= band[0]) & (freqs <= band[1])
    idx_total = (freqs >= 1) & (freqs <= 40)

    power_band = np.trapezoid(psd[idx_band], freqs[idx_band])
    power_total = np.trapezoid(psd[idx_total], freqs[idx_total])

    if power_total <= 0:
        return 0.0
    return float(power_band / power_total)


def extract_features_from_window(t7_window, t8_window):
    """
    Extrae exactamente las mismas features que se usaron en entrenamiento.
    """

    eps = 1e-8

    t7_window = np.asarray(t7_window, dtype=float)
    t8_window = np.asarray(t8_window, dtype=float)

    # Potencias de banda
    T7_theta_values = [band_power(t7_window, FS, THETA_BAND)]
    T7_mu_values = [band_power(t7_window, FS, MU_BAND)]
    T7_beta_values = [band_power(t7_window, FS, BETA_BAND)]

    T8_theta_values = [band_power(t8_window, FS, THETA_BAND)]
    T8_mu_values = [band_power(t8_window, FS, MU_BAND)]
    T8_beta_values = [band_power(t8_window, FS, BETA_BAND)]

    # Como en tiempo real calculamos una ventana, el "mean" es la potencia de esa ventana.
    # El "std" dentro de esta única ventana lo ponemos a 0.0 para mantener las 23 features.
    T7_theta = T7_theta_values[0]
    T7_mu = T7_mu_values[0]
    T7_beta = T7_beta_values[0]

    T8_theta = T8_theta_values[0]
    T8_mu = T8_mu_values[0]
    T8_beta = T8_beta_values[0]

    theta_diff = T7_theta - T8_theta
    mu_diff = T7_mu - T8_mu
    beta_diff = T7_beta - T8_beta

    theta_ratio = T7_theta / (T8_theta + eps)
    mu_ratio = T7_mu / (T8_mu + eps)
    beta_ratio = T7_beta / (T8_beta + eps)

    theta_asym = (T7_theta - T8_theta) / (T7_theta + T8_theta + eps)
    mu_asym = (T7_mu - T8_mu) / (T7_mu + T8_mu + eps)
    beta_asym = (T7_beta - T8_beta) / (T7_beta + T8_beta + eps)

    T7_mu_beta_ratio = T7_mu / (T7_beta + eps)
    T8_mu_beta_ratio = T8_mu / (T8_beta + eps)

    features = {
        "T7_theta_mean": T7_theta,
        "T7_mu_mean": T7_mu,
        "T7_beta_mean": T7_beta,

        "T8_theta_mean": T8_theta,
        "T8_mu_mean": T8_mu,
        "T8_beta_mean": T8_beta,

        "T7_theta_std": 0.0,
        "T7_mu_std": 0.0,
        "T7_beta_std": 0.0,

        "T8_theta_std": 0.0,
        "T8_mu_std": 0.0,
        "T8_beta_std": 0.0,

        "theta_diff_T7_T8": theta_diff,
        "mu_diff_T7_T8": mu_diff,
        "beta_diff_T7_T8": beta_diff,

        "theta_ratio_T7_T8": theta_ratio,
        "mu_ratio_T7_T8": mu_ratio,
        "beta_ratio_T7_T8": beta_ratio,

        "theta_asym_T7_T8": theta_asym,
        "mu_asym_T7_T8": mu_asym,
        "beta_asym_T7_T8": beta_asym,

        "T7_mu_beta_ratio": T7_mu_beta_ratio,
        "T8_mu_beta_ratio": T8_mu_beta_ratio
    }

    X = pd.DataFrame([features])
    X = X[feature_cols]

    return X.values


# ============================================================
# 5. LOOP DE TIEMPO REAL
# ============================================================

def main():
    print("\nConectando con Cortex...")
    ws = websocket.create_connection(
        CORTEX_URL,
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )

    cortex_token = authorize(ws)
    headset_id = query_headsets(ws)
    session_id = create_session(ws, cortex_token, headset_id)
    subscribe_eeg(ws, cortex_token, session_id)

    t7_buffer = deque(maxlen=WINDOW_SAMPLES)
    t8_buffer = deque(maxlen=WINDOW_SAMPLES)

    last_predictions = deque(maxlen=N_LAST_PREDICTIONS)

    eeg_labels = None
    sample_counter = 0

    print("\n================================================")
    print("TIEMPO REAL INICIADO")
    print("Piensa en mano DERECHA o IZQUIERDA.")
    print("Pulsa Ctrl+C para parar.")
    print("================================================\n")

    try:
        while True:
            message = json.loads(ws.recv())

            # Cortex puede enviar etiquetas del stream
            if "eeg" in message and "cols" in message:
                eeg_labels = message["cols"]
                print("Etiquetas EEG recibidas:", eeg_labels)
                continue

            if "eeg" not in message:
                continue

            eeg_data = message["eeg"]

            # Si todavía no tenemos etiquetas, usamos la estructura típica:
            # ["COUNTER","INTERPOLATED","AF3","T7","Pz","T8","AF4","RAW_CQ","MARKER_HARDWARE"]
            if eeg_labels is None:
                eeg_labels = [
                    "COUNTER", "INTERPOLATED", "AF3", "T7", "Pz", "T8", "AF4",
                    "RAW_CQ", "MARKER_HARDWARE"
                ]

            data_dict = dict(zip(eeg_labels, eeg_data))

            if "T7" not in data_dict or "T8" not in data_dict:
                print("No se encuentran T7/T8 en los datos recibidos.")
                continue

            t7_value = float(data_dict["T7"])
            t8_value = float(data_dict["T8"])

            t7_buffer.append(t7_value)
            t8_buffer.append(t8_value)

            sample_counter += 1

            # Esperar a llenar la ventana
            if len(t7_buffer) < WINDOW_SAMPLES:
                continue

            # Predecir cada cierto número de muestras
            if sample_counter % PREDICT_EVERY_SAMPLES != 0:
                continue

            X_new = extract_features_from_window(
                list(t7_buffer),
                list(t8_buffer)
            )

            pred_encoded = model.predict(X_new)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            last_predictions.append(pred_label)

            # Voto mayoritario
            final_prediction = Counter(last_predictions).most_common(1)[0][0]

            # Umbral mínimo de confianza para mostrar la predicción
            MIN_CONFIDENCE = 0.70

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_new)[0]
                prob_pairs = {
                    clase: prob for clase, prob in zip(label_encoder.classes_, probs)
                }
                max_confidence = max(prob_pairs.values())
                prob_text = " | " + " ".join(
                    [f"{k}: {v:.2f}" for k, v in prob_pairs.items()]
                )
                if max_confidence >= MIN_CONFIDENCE:
                    print(
                        f"✅ Predicción: {pred_label} "
                        f"→ Decisión final: {final_prediction}"
                        f"{prob_text}"
                    )
                # Si no supera el umbral, no imprime nada

            print(
                f"Predicción instantánea: {pred_label} "
                f"→ Decisión final: {final_prediction}"
                # si tenemos dos izquierda, luego derecha y luego dos izquierda, indica que todo es derecha por algun posible error
                f"{prob_text}"
            )

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
