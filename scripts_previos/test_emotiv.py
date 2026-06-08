import websocket

url = "wss://localhost:6868"

ws = websocket.create_connection(url)
print("Connected to Cortex")
ws.close()
