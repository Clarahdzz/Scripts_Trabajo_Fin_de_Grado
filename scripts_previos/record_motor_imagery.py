import websocket
import json
import ssl
import time
import csv
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================
CLIENT_ID = # tu id de cliente
CLIENT_SECRET = # tu id secreto

URL = "wss://localhost:6868"

OUTPUT_FILE = "eeg_motor_imagery_entrenamiento_v2.csv"

NUM_TRIALS = 10           # número de bloques de motor imagery
REST_SEC = 5.0         # duración reposo
ACTION_SEC = 5.0        # duración motor imagery
PREP_SEC = 3         # cuenta atrás inicial

REST_LABEL = 0
ACTION_LABEL = 1

REST_STATE = "rest"
ACTION_STATE = "motor_imagery_right_hand"

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def send_request(ws, payload):
    ws.send(json.dumps(payload))
    while True:
        response = ws.recv()
        data = json.loads(response)

        if "id" in data and data["id"] == payload["id"]:
            return data

def connect_cortex():
    return websocket.create_connection(
        URL,
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )

def authorize(ws):
    payload = {
        "jsonrpc": "2.0",
        "method": "authorize",
        "params": {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET,
            "debit": 5
        },
        "id": 1
    }
    result = send_request(ws, payload)

    if "error" in result:
        raise RuntimeError(f"Error en authorize: {result['error']}")

    return result["result"]["cortexToken"]

def query_headset(ws):
    payload = {
        "jsonrpc": "2.0",
        "method": "queryHeadsets",
        "params": {},
        "id": 2
    }
    result = send_request(ws, payload)

    if "error" in result:
        raise RuntimeError(f"Error en queryHeadsets: {result['error']}")

    if not result.get("result"):
        raise RuntimeError("No se ha detectado ningún headset.")

    return result["result"][0]["id"]

def get_or_create_session(ws, cortex_token, headset_id):
    # Buscar sesión existente
    query_sessions_payload = {
        "jsonrpc": "2.0",
        "method": "querySessions",
        "params": {
            "cortexToken": cortex_token
        },
        "id": 3
    }

    sessions_result = send_request(ws, query_sessions_payload)

    if "error" in sessions_result:
        raise RuntimeError(f"Error en querySessions: {sessions_result['error']}")

    for session in sessions_result.get("result", []):
        headset_info = session.get("headset", {})
        if headset_info.get("id") == headset_id and session.get("status") in ["active", "activated", "opened"]:
            return session["id"]

    # Si no existe, conectar y crear
    control_device_payload = {
        "jsonrpc": "2.0",
        "method": "controlDevice",
        "params": {
            "command": "connect",
            "headset": headset_id
        },
        "id": 4
    }
    control_result = send_request(ws, control_device_payload)

    if "error" in control_result:
        raise RuntimeError(f"Error en controlDevice: {control_result['error']}")

    time.sleep(3)

    create_session_payload = {
        "jsonrpc": "2.0",
        "method": "createSession",
        "params": {
            "cortexToken": cortex_token,
            "headset": headset_id,
            "status": "active"
        },
        "id": 5
    }

    session_result = send_request(ws, create_session_payload)

    if "error" in session_result:
        raise RuntimeError(f"Error en createSession: {session_result['error']}")

    return session_result["result"]["id"]

def subscribe_eeg(ws, cortex_token, session_id):
    payload = {
        "jsonrpc": "2.0",
        "method": "subscribe",
        "params": {
            "cortexToken": cortex_token,
            "session": session_id,
            "streams": ["eeg"]
        },
        "id": 6
    }

    result = send_request(ws, payload)

    if "error" in result:
        raise RuntimeError(f"Error en subscribe: {result['error']}")

def countdown(seconds):
    print("\n" + "=" * 60)
    print("EXPERIMENTO DE MOTOR IMAGERY")
    print("=" * 60)
    print("Instrucciones:")
    print("- Reposo: relájate, mira un punto fijo, no imagines movimiento.")
    print("- Acción: imagina cerrar el puño derecho con fuerza, SIN mover la mano.")
    print("- Evita mover cabeza, ojos y mandíbula.")
    print("=" * 60)

    seconds = int(seconds)

    for i in range(seconds, 0, -1):
        print(f"Comenzamos en {i}...")
        time.sleep(1)
        
def get_state_from_elapsed(elapsed_sec):
    """
    Devuelve:
    - label
    - state
    - trial
    - finished
    """
    cycle_sec = REST_SEC + ACTION_SEC
    total_experiment_sec = NUM_TRIALS * cycle_sec

    if elapsed_sec >= total_experiment_sec:
        return None, None, None, True

    trial_idx = int(elapsed_sec // cycle_sec) + 1
    within_cycle = elapsed_sec % cycle_sec

    if within_cycle < REST_SEC:
        return REST_LABEL, REST_STATE, trial_idx, False
    else:
        return ACTION_LABEL, ACTION_STATE, trial_idx, False

# =========================================================
# MAIN
# =========================================================
def main():
    if CLIENT_ID == "TU_CLIENT_ID" or CLIENT_SECRET == "TU_CLIENT_SECRET":
        raise ValueError("Debes rellenar CLIENT_ID y CLIENT_SECRET antes de ejecutar.")

    output_path = Path(OUTPUT_FILE)

    ws = None
    try:
        ws = connect_cortex()
        print("Conectado a Cortex")

        cortex_token = authorize(ws)
        print("Autorización correcta")

        headset_id = query_headset(ws)
        print(f"Headset detectado: {headset_id}")

        session_id = get_or_create_session(ws, cortex_token, headset_id)
        print(f"Sesión activa: {session_id}")

        subscribe_eeg(ws, cortex_token, session_id)
        print("Suscripción EEG correcta")

        countdown(PREP_SEC)

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "sample",
                "AF3",
                "T7",
                "Pz",
                "T8",
                "AF4",
                "marker_hardware",
                "label",
                "trial",
                "state"
            ])

            exp_start = time.monotonic()
            last_state = None
            last_trial = None

            print("\n✅ Grabación iniciada\n")

            while True:
                msg = ws.recv()
                data = json.loads(msg)

                if "eeg" not in data:
                    continue

                elapsed = time.monotonic() - exp_start
                label, state, trial, finished = get_state_from_elapsed(elapsed)

                if finished:
                    print("\n✅ Experimento terminado con éxito.")
                    break

                if state != last_state or trial != last_trial:
                    print("\a", end="")
                    if state == REST_STATE:
                        print(f"🟢 Trial {trial}/{NUM_TRIALS} - REPOSO")
                    else:
                        print(f"🔴 Trial {trial}/{NUM_TRIALS} - MOTOR IMAGERY (mano derecha)")
                    last_state = state
                    last_trial = trial

                eeg = data["eeg"]

                writer.writerow([
                    data.get("time", ""),
                    eeg[0] if len(eeg) > 0 else "",
                    eeg[2] if len(eeg) > 2 else "",
                    eeg[3] if len(eeg) > 3 else "",
                    eeg[4] if len(eeg) > 4 else "",
                    eeg[5] if len(eeg) > 5 else "",
                    eeg[6] if len(eeg) > 6 else "",
                    eeg[7] if len(eeg) > 7 else "",
                    label,
                    trial,
                    state
                ])

        print(f"Archivo guardado en: {output_path.resolve()}")

    except KeyboardInterrupt:
        print("\n⛔ Grabación detenida por el usuario.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        if ws is not None:
            ws.close()
            print("Conexión cerrada.")

if __name__ == "__main__":
    main()
