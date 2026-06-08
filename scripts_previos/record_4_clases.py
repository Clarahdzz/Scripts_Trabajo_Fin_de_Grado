import websocket
import json
import ssl
import time
import csv
from pathlib import Path

# =========================================================
# CONFIGURACIÓN - RELLENA TUS CREDENCIALES
# =========================================================
CLIENT_ID     = # tu id de cliente
CLIENT_SECRET = # tu id secreto
URL           = "wss://localhost:6868"

OUTPUT_FILE   = "eeg_4clases_entrenamiento.csv"

# =========================================================
# PROTOCOLO DEL EXPERIMENTO
# =========================================================
# Etiquetas numéricas
LABEL_REPOSO        = 0
LABEL_MANO_CERRADA  = 1
LABEL_MANO_ABIERTA  = 2
LABEL_RUIDO         = 3

# Nombre de cada clase (para mostrar por pantalla)
CLASS_NAMES = {
    LABEL_REPOSO:       "REPOSO         (no hagas nada, relájate)",
    LABEL_MANO_CERRADA: "MANO CERRADA   (imagina cerrar el puño derecho)",
    LABEL_MANO_ABIERTA: "MANO ABIERTA   (imagina abrir la mano derecha)",
    LABEL_RUIDO:        "RUIDO/DISTRACT (mueve los ojos, traga saliva, piensa en otra cosa)",
}

# Duración de cada bloque (segundos)
DURATION = {
    LABEL_REPOSO:       6.0,
    LABEL_MANO_CERRADA: 6.0,
    LABEL_MANO_ABIERTA: 6.0,
    LABEL_RUIDO:        6.0,
}

# Orden de clases dentro de cada trial (puedes cambiar el orden)
CLASS_ORDER = [
    LABEL_REPOSO,
    LABEL_MANO_CERRADA,
    LABEL_REPOSO,
    LABEL_MANO_ABIERTA,
    LABEL_REPOSO,
    LABEL_RUIDO,
]

NUM_TRIALS  = 10    # repeticiones del bloque completo
PREP_SEC    = 5     # cuenta atrás inicial

# =========================================================
# FUNCIONES AUXILIARES CORTEX API
# =========================================================
def send_request(ws, payload):
    ws.send(json.dumps(payload))
    while True:
        response = ws.recv()
        data = json.loads(response)
        if "id" in data and data["id"] == payload["id"]:
            return data

def connect_cortex():
    return websocket.create_connection(URL, sslopt={"cert_reqs": ssl.CERT_NONE})

def authorize(ws):
    result = send_request(ws, {
        "jsonrpc": "2.0", "method": "authorize",
        "params": {"clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET, "debit": 5},
        "id": 1
    })
    if "error" in result:
        raise RuntimeError(f"Error en authorize: {result['error']}")
    return result["result"]["cortexToken"]

def query_headset(ws):
    result = send_request(ws, {
        "jsonrpc": "2.0", "method": "queryHeadsets",
        "params": {}, "id": 2
    })
    if not result.get("result"):
        raise RuntimeError("No se detectó ningún headset.")
    return result["result"][0]["id"]

def get_or_create_session(ws, cortex_token, headset_id):
    sessions = send_request(ws, {
        "jsonrpc": "2.0", "method": "querySessions",
        "params": {"cortexToken": cortex_token}, "id": 3
    })
    for s in sessions.get("result", []):
        if s["headset"]["id"] == headset_id and s["status"] in ["active", "activated", "opened"]:
            return s["id"]

    send_request(ws, {
        "jsonrpc": "2.0", "method": "controlDevice",
        "params": {"command": "connect", "headset": headset_id}, "id": 4
    })
    time.sleep(3)

    result = send_request(ws, {
        "jsonrpc": "2.0", "method": "createSession",
        "params": {"cortexToken": cortex_token, "headset": headset_id, "status": "active"},
        "id": 5
    })
    if "error" in result:
        raise RuntimeError(f"Error en createSession: {result['error']}")
    return result["result"]["id"]

def subscribe_eeg(ws, cortex_token, session_id):
    result = send_request(ws, {
        "jsonrpc": "2.0", "method": "subscribe",
        "params": {"cortexToken": cortex_token, "session": session_id, "streams": ["eeg"]},
        "id": 6
    })
    if "error" in result:
        raise RuntimeError(f"Error en subscribe: {result['error']}")

# =========================================================
# PROTOCOLO: genera la secuencia completa de bloques
# =========================================================
def build_sequence():
    """
    Devuelve lista de dicts con:
        label, class_name, start_sec, end_sec, trial
    """
    sequence = []
    t = 0.0
    for trial in range(1, NUM_TRIALS + 1):
        for label in CLASS_ORDER:
            duration = DURATION[label]
            sequence.append({
                "label":      label,
                "class_name": CLASS_NAMES[label],
                "start_sec":  t,
                "end_sec":    t + duration,
                "trial":      trial,
            })
            t += duration
    return sequence, t  # t = duración total del experimento

# =========================================================
# MAIN
# =========================================================
def main():
    sequence, total_sec = build_sequence()

    print("\n" + "=" * 60)
    print("  GRABACIÓN EEG - 4 CLASES")
    print("=" * 60)
    print(f"  Trials:          {NUM_TRIALS}")
    print(f"  Bloques/trial:   {len(CLASS_ORDER)}")
    print(f"  Duración bloque: {list(DURATION.values())[0]} s")
    print(f"  Duración total:  {total_sec:.0f} s  ({total_sec/60:.1f} min)")
    print(f"  Archivo salida:  {OUTPUT_FILE}")
    print("=" * 60)
    print("\nClases a grabar:")
    for label, name in CLASS_NAMES.items():
        print(f"  [{label}] {name}")
    print()

    ws = None
    try:
        ws = connect_cortex()
        print("Conectado a Cortex")

        cortex_token = authorize(ws)
        print("Autorización correcta")

        headset_id = query_headset(ws)
        print(f"Headset: {headset_id}")

        session_id = get_or_create_session(ws, cortex_token, headset_id)
        print(f"Sesión activa: {session_id}")

        subscribe_eeg(ws, cortex_token, session_id)
        print("Suscripción EEG correcta")

        # Cuenta atrás
        print()
        for i in range(PREP_SEC, 0, -1):
            print(f"  Comenzamos en {i}...")
            time.sleep(1)
        print("\n✅ ¡GRABACIÓN INICIADA!\n")

        with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "sample",
                "AF3", "T7", "Pz", "T8", "AF4",
                "marker_hardware",
                "label", "trial", "state"
            ])

            exp_start    = time.monotonic()
            current_idx  = 0
            last_printed = -1

            while True:
                msg  = ws.recv()
                data = json.loads(msg)

                if "eeg" not in data:
                    continue

                elapsed = time.monotonic() - exp_start

                # Avanzar al bloque correcto
                while current_idx < len(sequence) and elapsed >= sequence[current_idx]["end_sec"]:
                    current_idx += 1

                # Experimento terminado
                if current_idx >= len(sequence):
                    print("\n✅ Experimento terminado con éxito.")
                    break

                block = sequence[current_idx]

                # Mostrar aviso al cambiar de bloque
                if current_idx != last_printed:
                    print("\a", end="", flush=True)
                    t_left = block["end_sec"] - block["start_sec"]
                    print(f"  Trial {block['trial']:2d}/{NUM_TRIALS}  |  {block['class_name']}  ({t_left:.0f}s)")
                    last_printed = current_idx

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
                    block["label"],
                    block["trial"],
                    block["class_name"].split()[0].lower(),
                ])

        print(f"\n📁 Archivo guardado: {Path(OUTPUT_FILE).resolve()}")

    except KeyboardInterrupt:
        print("\n⛔ Grabación detenida por el usuario.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        if ws:
            ws.close()
            print("Conexión cerrada.")

if __name__ == "__main__":
    main()
