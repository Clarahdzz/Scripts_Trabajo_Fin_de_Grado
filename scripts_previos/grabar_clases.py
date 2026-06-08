import os
import ssl
import csv
import json
import time
import websocket
from pathlib import Path
from typing import Optional, Dict, Any, List

# =========================
# CONFIGURACIÓN
# =========================

CLIENT_ID     = # tu client id
CLIENT_SECRET = # tu id secreto

URL = "wss://localhost:6868"

MODE = "movimiento_real"      # "movimiento_real" o "imaginacion_motora"
LABEL = "mano_cerrada"              # "reposo", "mano_abierta", "mano_cerrada", "ruido"

NUM_TRIALS = 20
COUNTDOWN_SEC = 5
WINDOW_SECONDS = 3.0
REST_SECONDS = 3.0

# Ajusta esto si tu muestreo real es distinto
SAMPLING_RATE = 128
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)

CHANNEL_NAMES = ["AF3", "T7", "Pz", "T8", "AF4"]

BASE_DIR = Path("dataset") / MODE / LABEL
BASE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CORTEX API
# =========================================================

def send_request(ws, payload: Dict[str, Any]) -> Dict[str, Any]:
    ws.send(json.dumps(payload))
    while True:
        response = ws.recv()
        data = json.loads(response)
        if "id" in data and data["id"] == payload["id"]:
            return data

def connect_cortex():
    return websocket.create_connection(URL, sslopt={"cert_reqs": ssl.CERT_NONE})

def request_access(ws):
    result = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "requestAccess",
        "params": {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        },
        "id": 1
    })
    if "error" in result:
        raise RuntimeError(f"Error en requestAccess: {result['error']}")
    return result

def authorize(ws) -> str:
    result = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "authorize",
        "params": {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        },
        "id": 2
    })
    if "error" in result:
        raise RuntimeError(f"Error en authorize: {result['error']}")
    return result["result"]["cortexToken"]

def query_headset(ws) -> str:
    result = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "queryHeadsets",
        "params": {},
        "id": 3
    })
    if not result.get("result"):
        raise RuntimeError("No se detectó ningún headset.")
    return result["result"][0]["id"]

def get_or_create_session(ws, cortex_token: str, headset_id: str) -> str:
    sessions = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "querySessions",
        "params": {"cortexToken": cortex_token},
        "id": 4
    })

    for s in sessions.get("result", []):
        try:
            if s["headset"]["id"] == headset_id and s["status"] in ["active", "activated", "opened"]:
                return s["id"]
        except KeyError:
            pass

    send_request(ws, {
        "jsonrpc": "2.0",
        "method": "controlDevice",
        "params": {"command": "connect", "headset": headset_id},
        "id": 5
    })
    time.sleep(2)

    result = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "createSession",
        "params": {
            "cortexToken": cortex_token,
            "headset": headset_id,
            "status": "active"
        },
        "id": 6
    })
    if "error" in result:
        raise RuntimeError(f"Error en createSession: {result['error']}")
    return result["result"]["id"]

def subscribe_eeg(ws, cortex_token: str, session_id: str):
    result = send_request(ws, {
        "jsonrpc": "2.0",
        "method": "subscribe",
        "params": {
            "cortexToken": cortex_token,
            "session": session_id,
            "streams": ["eeg"]
        },
        "id": 7
    })
    if "error" in result:
        raise RuntimeError(f"Error en subscribe: {result['error']}")

# =========================================================
# PARSEO EEG
# =========================================================

def parse_eeg_packet(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ajusta esta función si tu formato real de eeg cambia.
    Aquí asumimos algo parecido a:
    eeg[0] = sample
    eeg[2:7] = AF3, T7, Pz, T8, AF4
    eeg[7] = marker_hardware
    """
    eeg = data.get("eeg")
    if not isinstance(eeg, list):
        return None

    if len(eeg) < 8:
        return None

    try:
        sample_idx = int(eeg[0])
        channels = [float(eeg[2]), float(eeg[3]), float(eeg[4]), float(eeg[5]), float(eeg[6])]
        marker_hw = eeg[7]
        timestamp = data.get("time", time.time())
    except (ValueError, TypeError, IndexError):
        return None

    return {
        "time": timestamp,
        "sample": sample_idx,
        "AF3": channels[0],
        "T7": channels[1],
        "Pz": channels[2],
        "T8": channels[3],
        "AF4": channels[4],
        "marker_hardware": marker_hw,
    }

# =========================================================
# GUARDADO
# =========================================================

def next_trial_filename(trial_num: int) -> Path:
    return BASE_DIR / f"{LABEL}_{trial_num:03d}.csv"

def save_trial_csv(rows: List[Dict[str, Any]], trial_num: int):
    file_path = next_trial_filename(trial_num)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "sample",
            "AF3", "T7", "Pz", "T8", "AF4",
            "marker_hardware",
            "label", "trial", "state", "mode"
        ])
        for row in rows:
            writer.writerow([
                row["time"],
                row["sample"],
                row["AF3"], row["T7"], row["Pz"], row["T8"], row["AF4"],
                row["marker_hardware"],
                LABEL,
                trial_num,
                LABEL,
                MODE
            ])
    print(f"✅ Guardado: {file_path}")

# =========================================================
# EXPERIMENTO
# =========================================================

def countdown():
    print("\nPrepárate...")
    for i in range(COUNTDOWN_SEC, 0, -1):
        print(f"⏳ {i}...")
        time.sleep(1)
    print("🚀 ¡YA!")

def collect_one_trial(ws) -> List[Dict[str, Any]]:
    rows = []

    start_time = time.time()
    last_print = -1

    print("🔴 GRABANDO...")

    while len(rows) < WINDOW_SAMPLES:
        msg = ws.recv()
        data = json.loads(msg)
        parsed = parse_eeg_packet(data)

        if parsed is not None:
            rows.append(parsed)

        # Mostrar cuenta atrás durante grabación
        elapsed = time.time() - start_time
        remaining = WINDOW_SECONDS - elapsed

        # Mostrar cada ~0.5s para no saturar consola
        if int(remaining * 10) != last_print:
            last_print = int(remaining * 10)

            if remaining > 0:
                print(f"⏱️ Grabando... {remaining:.1f}s restantes", end="\r")
            else:
                print("⏱️ Grabación terminada            ")

    print("✔ Ventana completa")

    return rows

def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Faltan credenciales. Define EMOTIV_CLIENT_ID y EMOTIV_CLIENT_SECRET "
            "como variables de entorno."
        )

    print("=" * 60)
    print("CAPTURA EEG POR ENSAYOS")
    print("=" * 60)
    print(f"Modo: {MODE}")
    print(f"Clase: {LABEL}")
    print(f"Ensayos: {NUM_TRIALS}")
    print(f"Cuenta atrás: {COUNTDOWN_SEC}s")
    print(f"Duración grabación: {WINDOW_SECONDS}s")
    print(f"Descanso: {REST_SECONDS}s")
    print(f"Salida: {BASE_DIR.resolve()}")
    print("=" * 60)

    ws = None
    try:
        ws = connect_cortex()
        print("✅ Conectado a Cortex")

        request_access(ws)
        cortex_token = authorize(ws)
        print("✅ Autorización correcta")

        headset_id = query_headset(ws)
        print(f"✅ Headset detectado: {headset_id}")

        session_id = get_or_create_session(ws, cortex_token, headset_id)
        print(f"✅ Sesión activa: {session_id}")

        subscribe_eeg(ws, cortex_token, session_id)
        print("✅ Suscripción EEG correcta")

        for trial in range(1, NUM_TRIALS + 1):
            print()
            print(f"========== ENSAYO {trial}/{NUM_TRIALS} | {LABEL.upper()} ==========")
            print("Prepárate")
            countdown()

            t0 = time.time()
            rows = collect_one_trial(ws)
            dt = time.time() - t0
            print(f"⏱️ Captura completada en {dt:.2f}s")

            save_trial_csv(rows, trial)

            if trial < NUM_TRIALS:
                print(f"😌 Descanso {REST_SECONDS}s")
                time.sleep(REST_SECONDS)

        print("\n✅ Captura finalizada correctamente")

    except KeyboardInterrupt:
        print("\n⛔ Captura detenida por el usuario")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        if ws:
            ws.close()
            print("🔌 Conexión cerrada")

if __name__ == "__main__":
    main()
