import websocket
import json
import ssl
import time
import csv

CLIENT_ID = # tu id de cliente
CLIENT_SECRET = # tu id secreto

URL = "wss://localhost:6868"


def send_request(ws, payload):
    ws.send(json.dumps(payload))
    while True:
        response = ws.recv()
        data = json.loads(response)

        if "id" in data and data["id"] == payload["id"]:
            return data


ws = websocket.create_connection(
    URL,
    sslopt={"cert_reqs": ssl.CERT_NONE}
)

print("Conectado a Cortex")

# 1) Authorize
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

# 2) Query headsets
query_headsets_payload = {
    "jsonrpc": "2.0",
    "method": "queryHeadsets",
    "params": {},
    "id": 2
}

headsets_result = send_request(ws, query_headsets_payload)
headset_id = headsets_result["result"][0]["id"]

# 3) Query sessions
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
        break

# 4) If no session exists, create one
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
    session_id = session_result["result"]["id"]

# 5) Subscribe to EEG
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

send_request(ws, subscribe_payload)

# 6) Save EEG to CSV
filename = "eeg_ojos_abiertos_cerrados.csv"

with open(filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "sample", "AF3", "T7", "Pz", "T8", "AF4", "marker_hardware"])

    print(f"Guardando datos en {filename} ...")
    print("Pulsa Ctrl+C para detener.")

    try:
        while True:
            msg = ws.recv()
            data = json.loads(msg)

            if "eeg" in data:
                eeg = data["eeg"]
                timestamp = data["time"]

                sample = eeg[0]
                af3 = eeg[2]
                t7 = eeg[3]
                pz = eeg[4]
                t8 = eeg[5]
                af4 = eeg[6]
                marker_hardware = eeg[7]

                writer.writerow([timestamp, sample, af3, t7, pz, t8, af4, marker_hardware])

    except KeyboardInterrupt:
        print("\nGrabación detenida por el usuario.")

ws.close()
print("Conexión cerrada.")
