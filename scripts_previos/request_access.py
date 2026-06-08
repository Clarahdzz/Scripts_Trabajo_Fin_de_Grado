import websocket
import json
import ssl

CLIENT_ID = # tu id de cliente
CLIENT_SECRET = # tu id secreto

URL = "wss://localhost:6868"

def send_request(ws, payload):
    ws.send(json.dumps(payload))
    response = ws.recv()
    print("Respuesta:")
    print(response)
    print("-" * 50)
    return json.loads(response)

ws = websocket.create_connection(
    URL,
    sslopt={"cert_reqs": ssl.CERT_NONE}
)

print("Conectado a Cortex")

request_access_payload = {
    "jsonrpc": "2.0",
    "method": "requestAccess",
    "params": {
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET
    },
    "id": 1
}

send_request(ws, request_access_payload)

ws.close()
print("Conexión cerrada")
