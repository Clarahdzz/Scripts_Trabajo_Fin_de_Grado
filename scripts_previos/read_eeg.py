import websocket
import json
import ssl
import time

CLIENT_ID = # tu id de cliente
CLIENT_SECRET = # tu id secreto

URL = "wss://localhost:6868"


def send_request(ws, payload):
    ws.send(json.dumps(payload))
    while True:
        response = ws.recv()
        data = json.loads(response)

        # Solo devolver la respuesta con el mismo id que hemos enviado
        if "id" in data and data["id"] == payload["id"]:
            print("RESPUESTA:")
            print(response)
            print("-" * 60)
            return data
        else:
            print("MENSAJE ADICIONAL:")
            print(response)
            print("-" * 60)


# Conectar con Cortex
ws = websocket.create_connection(
    URL,
    sslopt={"cert_reqs": ssl.CERT_NONE}
)

print("Conectado a Cortex")
print("=" * 60)

# 1) AUTHORIZE
authorize_payload = {
    "jsonrpc": "2.0",
    "method": "authorize",
    "params": {
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET,
        "debit": 5
    },
    "id": 1
}

auth_result = send_request(ws, authorize_payload)
cortex_token = auth_result["result"]["cortexToken"]

print("TOKEN OBTENIDO")
print("=" * 60)

# 2) QUERY HEADSETS
query_headsets_payload = {
    "jsonrpc": "2.0",
    "method": "queryHeadsets",
    "params": {},
    "id": 2
}

headsets_result = send_request(ws, query_headsets_payload)

if not headsets_result["result"]:
    print("No se ha detectado ningún headset.")
    ws.close()
    exit()

headset_id = headsets_result["result"][0]["id"]
print(f"Headset detectado: {headset_id}")
print("=" * 60)

# 3) QUERY SESSIONS -> ver si ya hay una sesión activa
query_sessions_payload = {
    "jsonrpc": "2.0",
    "method": "querySessions",
    "params": {
        "cortexToken": cortex_token
    },
    "id": 3
}

sessions_result = send_request(ws, query_sessions_payload)

session_id = None
for session in sessions_result.get("result", []):
    if session["headset"]["id"] == headset_id and session["status"] in ["active", "activated", "opened"]:
        session_id = session["id"]
        print(f"Sesión ya existente encontrada: {session_id}")
        print("=" * 60)
        break

# 4) Si no existe sesión, conectar headset y crearla
if session_id is None:
    control_device_payload = {
        "jsonrpc": "2.0",
        "method": "controlDevice",
        "params": {
            "command": "connect",
            "headset": headset_id
        },
        "id": 4
    }

    send_request(ws, control_device_payload)

    print("Esperando unos segundos para que conecte el headset...")
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
        print("Error al crear la sesión:")
        print(session_result["error"])
        ws.close()
        exit()

    session_id = session_result["result"]["id"]
    print(f"Sesión creada: {session_id}")
    print("=" * 60)

# 5) SUBSCRIBE EEG
subscribe_payload = {
    "jsonrpc": "2.0",
    "method": "subscribe",
    "params": {
        "cortexToken": cortex_token,
        "session": session_id,
        "streams": ["eeg"]
    },
    "id": 6
}

subscribe_result = send_request(ws, subscribe_payload)

if "error" in subscribe_result:
    print("Error al suscribirse al EEG:")
    print(subscribe_result["error"])
    ws.close()
    exit()

print("Recibiendo datos EEG en tiempo real...")
print("Pulsa Ctrl+C para detener.")
print("=" * 60)

try:
    while True:
        data = ws.recv()
        print(data)

except KeyboardInterrupt:
    print("\nLectura detenida por el usuario.")

finally:
    ws.close()
    print("Conexión cerrada.")
