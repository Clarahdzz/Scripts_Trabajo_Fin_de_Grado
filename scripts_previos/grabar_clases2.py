import websocket
import json
import ssl
import time
import csv
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================
CLIENT_ID     = # tu id de cliente
CLIENT_SECRET = # tu id secreto
URL           = "wss://localhost:6868"

# Frecuencia de muestreo del Emotiv Insight
FS            = 128          # Hz

# Duración de cada muestra grabada (segundos)
SAMPLE_SEC    = 5.0
SAMPLES_PER_FILE = int(SAMPLE_SEC * FS)   # ~384 muestras por fichero

# Cuenta atrás entre muestras (segundos)
COUNTDOWN_SEC = 5

# Número de ejemplos por clase
NUM_EXAMPLES  = 25

# Carpeta de salida
OUTPUT_DIR    = "eeg_edge_impulse_dataset"

# Clases a grabar: (nombre_label, instrucción para el usuario)
CLASSES = [
    ("reposo", "No hagas nada. Relájate, mira al frente, mente en blanco."),
    ("imagery_derecha", "Imagina cerrar el puño DERECHO repetidamente. No muevas la mano."),
    ("imagery_izquierda", "Imagina cerrar el puño IZQUIERDO repetidamente. No muevas la mano."),
]

# Canales del Emotiv Insight (índices en el array EEG de Cortex API)
# eeg[0]=sample, eeg[1]=?, eeg[2]=AF3, eeg[3]=T7, eeg[4]=Pz, eeg[5]=T8, eeg[6]=AF4
CHANNEL_NAMES   = ["AF3", "T7", "Pz", "T8", "AF4"]
CHANNEL_INDICES = [2,      3,    4,    5,     6   ]


# =========================================================
# FUNCIONES CORTEX API
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
            print(f"  Sesión existente reutilizada: {s['id']}")
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
# FUNCIONES DE INTERFAZ EN TERMINAL
# =========================================================
def print_separator():
    print("=" * 60)

def print_header():
    print_separator()
    print("  GRABACIÓN EEG PARA EDGE IMPULSE")
    print("  Emotiv Insight — 4 clases")
    print_separator()
    print(f"  Muestras por clase : {NUM_EXAMPLES}")
    print(f"  Duración muestra   : {SAMPLE_SEC} s  ({SAMPLES_PER_FILE} muestras a {FS} Hz)")
    print(f"  Cuenta atrás       : {COUNTDOWN_SEC} s")
    print(f"  Canales grabados   : {', '.join(CHANNEL_NAMES)}")
    print(f"  Carpeta salida     : {OUTPUT_DIR}/")
    print_separator()
    print()

def countdown(label, example_num, total, instruction):
    """Muestra la instrucción y cuenta atrás antes de grabar."""
    print()
    print_separator()
    print(f"  CLASE     : {label.upper()}")
    print(f"  EJEMPLO   : {example_num} / {total}")
    print(f"  ACCIÓN    : {instruction}")
    print_separator()
    print()
    for i in range(COUNTDOWN_SEC, 0, -1):
        print(f"  Preparando... {i}", end="\r", flush=True)
        time.sleep(1)
    print(f"  {'⏺  GRABANDO ' + str(SAMPLE_SEC) + 's...':50}", flush=True)

def show_progress(current, total):
    """Barra de progreso simple en terminal."""
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * current / total
    print(f"  [{bar}] {current:3d}/{total}  ({pct:.0f}%)", end="\r", flush=True)


# =========================================================
# GRABACIÓN DE UNA MUESTRA
# =========================================================
def record_sample(ws, label, example_num):
    """
    Graba SAMPLES_PER_FILE muestras EEG y las guarda en formato Edge Impulse.
    Nombre fichero: <label>.<example_num:02d>.csv
    Columnas: timestamp, AF3, T7, Pz, T8, AF4
    timestamp en ms, empieza en 0, incremento = MS_PER_SAMPLE
    """
    filename = f"{label}.{example_num:02d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    rows = []
    timestamp_ms = 0

    while len(rows) < SAMPLES_PER_FILE:
        msg  = ws.recv()
        data = json.loads(msg)

        if "eeg" not in data:
            continue

        eeg = data["eeg"]

        row = [timestamp_ms]
        for idx in CHANNEL_INDICES:
            row.append(eeg[idx] if len(eeg) > idx else 0.0)

        rows.append(row)
        timestamp_ms = int(round(len(rows) * 1000 / FS))

        # Mostrar progreso
        show_progress(len(rows), SAMPLES_PER_FILE)

    # Salto de línea tras la barra de progreso
    print()

    # Guardar CSV
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + CHANNEL_NAMES)
        writer.writerows(rows)

    print(f"  ✅ Guardado: {filepath}  ({len(rows)} muestras)")
    return filepath


# =========================================================
# MAIN
# =========================================================
def main():
    print_header()

    # Crear carpeta de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ws = None
    try:
        # --- Conexión a Cortex ---
        print("  Conectando con Emotiv Cortex...")
        ws = connect_cortex()
        print("  ✅ Conectado a Cortex")

        cortex_token = authorize(ws)
        print("  ✅ Autorización correcta")

        headset_id = query_headset(ws)
        print(f"  ✅ Headset detectado: {headset_id}")

        session_id = get_or_create_session(ws, cortex_token, headset_id)
        print(f"  ✅ Sesión activa: {session_id}")

        subscribe_eeg(ws, cortex_token, session_id)
        print("  ✅ Stream EEG activo")
        print()

        # --- Resumen del plan de grabación ---
        total_files  = len(CLASSES) * NUM_EXAMPLES
        total_min    = total_files * (SAMPLE_SEC + COUNTDOWN_SEC) / 60
        print(f"  Total ficheros a grabar : {total_files}")
        print(f"  Tiempo estimado         : {total_min:.1f} minutos")
        print()
        input("  Pulsa ENTER cuando estés listo para empezar...")

        # --- Bucle principal de grabación ---
        files_recorded = []

        for label, instruction in CLASSES:
            print()
            print_separator()
            print(f"  INICIANDO CLASE: {label.upper()}")
            print(f"  {instruction}")
            print_separator()
            input(f"  Pulsa ENTER para empezar a grabar '{label}'...")

            for example_num in range(1, NUM_EXAMPLES + 1):
                countdown(label, example_num, NUM_EXAMPLES, instruction)
                path = record_sample(ws, label, example_num)
                files_recorded.append(path)

            print()
            print(f"  ✅ Clase '{label}' completada ({NUM_EXAMPLES} ficheros)")

        # --- Resumen final ---
        print()
        print_separator()
        print("  GRABACIÓN COMPLETADA")
        print_separator()
        print(f"  Ficheros grabados : {len(files_recorded)}")
        print(f"  Carpeta           : {os.path.abspath(OUTPUT_DIR)}/")
        print()
        print("  PRÓXIMOS PASOS EN EDGE IMPULSE:")
        print("  1. Ve a tu proyecto → Data acquisition")
        print("  2. Haz clic en 'Upload existing data'")
        print(f"  3. Selecciona todos los CSV de la carpeta '{OUTPUT_DIR}/'")
        print("  4. El label se asigna automáticamente por el nombre del fichero")
        print("     (reposo.01.csv → label 'reposo', etc.)")
        print_separator()

    except KeyboardInterrupt:
        print("\n\n  ⛔ Grabación interrumpida por el usuario.")
        if files_recorded:
            print(f"  Se grabaron {len(files_recorded)} ficheros antes de interrumpir.")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")

    finally:
        if ws:
            ws.close()
            print("  Conexión cerrada.")


if __name__ == "__main__":
    main()
