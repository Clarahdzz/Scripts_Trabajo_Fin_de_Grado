import asyncio
import json
import ssl
from typing import Any, Dict, Optional

import websockets

CLIENT_ID     = # tu id de cliente
CLIENT_SECRET = # tu id secreto
URL           = "wss://localhost:6868"

# Nombre exacto del profile que ya entrenaste en EMOTIV
PROFILE_NAME = "TFG 2"

# Umbral mínimo de confianza para mostrar un comando
MIN_SCORE = 0.30


class CortexComReader:
    def __init__(self, client_id: str, client_secret: str, profile_name: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.profile_name = profile_name

        self.ws = None
        self.req_id = 1

        self.cortex_token: Optional[str] = None
        self.headset_id: Optional[str] = None
        self.session_id: Optional[str] = None

        self.com_cols = []

    async def rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self.req_id,
            "method": method,
            "params": params,
        }
        self.req_id += 1

        await self.ws.send(json.dumps(payload))

        while True:
            raw = await self.ws.recv()
            msg = json.loads(raw)

            if "id" in msg and msg["id"] == payload["id"]:
                if "error" in msg:
                    raise RuntimeError(f"{method} -> {msg['error']}")
                return msg["result"]

            # Ignoramos mensajes asíncronos aquí; los leeremos luego en el loop principal

    async def connect(self):
        ssl_ctx = ssl._create_unverified_context()
        self.ws = await websockets.connect(URL, ssl=ssl_ctx)
        print("✅ Conectado a Cortex")

    async def request_access(self):
        result = await self.rpc(
            "requestAccess",
            {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
            },
        )
        granted = result.get("accessGranted", False)
        print(f"requestAccess -> accessGranted={granted}")
        if not granted:
            raise RuntimeError(
                "La app no tiene permiso. Abre EMOTIV Launcher/App y aprueba el acceso."
            )

    async def refresh_and_find_headset(self):
        # El flujo oficial menciona refresh antes de queryHeadsets cuando haga falta. :contentReference[oaicite:4]{index=4}
        try:
            await self.rpc("controlDevice", {"command": "refresh"})
            await asyncio.sleep(2)
        except Exception:
            # Si no hace falta, no pasa nada
            pass

        result = await self.rpc("queryHeadsets", {})
        if not result:
            raise RuntimeError("No se detectó ningún headset.")

        self.headset_id = result[0]["id"]
        print(f"✅ Headset detectado: {self.headset_id}")

    async def connect_headset(self):
        await self.rpc(
            "controlDevice",
            {
                "command": "connect",
                "headset": self.headset_id,
            },
        )
        await asyncio.sleep(3)
        print("✅ Headset conectado")

    async def authorize(self):
        result = await self.rpc(
            "authorize",
            {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
                "debit": 1,
            },
        )
        self.cortex_token = result["cortexToken"]
        print("✅ Autorización correcta")

    async def create_session(self):
        result = await self.rpc(
            "createSession",
            {
                "cortexToken": self.cortex_token,
                "headset": self.headset_id,
                "status": "active",
            },
        )
        self.session_id = result["id"]
        print(f"✅ Sesión activa: {self.session_id}")

    async def load_profile(self):
        # setupProfile load es el método oficial para cargar el perfil en el headset. :contentReference[oaicite:5]{index=5}
        result = await self.rpc(
            "setupProfile",
            {
                "cortexToken": self.cortex_token,
                "headset": self.headset_id,
                "profile": self.profile_name,
                "status": "load",
            },
        )
        print(f"✅ Profile cargado: {result.get('name', self.profile_name)}")

    async def subscribe_com(self):
        result = await self.rpc(
            "subscribe",
            {
                "cortexToken": self.cortex_token,
                "session": self.session_id,
                "streams": ["com"],
            },
        )

        success_streams = result.get("success", [])
        if not success_streams:
            raise RuntimeError(f"No se pudo suscribir a 'com': {result}")

        com_info = None
        for stream in success_streams:
            if stream.get("streamName") == "com":
                com_info = stream
                break

        if com_info is None:
            raise RuntimeError("La suscripción a 'com' no devolvió información de columnas.")

        self.com_cols = com_info.get("cols", [])
        print(f"✅ Suscrito a 'com' con columnas: {self.com_cols}")

    def parse_com_sample(self, msg: Dict[str, Any]):
        """
        Interpreta el stream 'com' usando las columnas devueltas por subscribe.
        El ejemplo oficial muestra cols = ['act', 'pow']. :contentReference[oaicite:6]{index=6}
        """
        values = msg.get("com")
        if not values or not self.com_cols:
            return None

        sample = dict(zip(self.com_cols, values))

        cmd = sample.get("act", "")
        score = sample.get("pow", 0.0)

        label_map = {
            "push": "derecha",
            "pull": "izquierda",
            "neutral": "reposo",
        }

        if score is None:
            score = 0.0

        label = label_map.get(cmd, "desconocido")
        return cmd, float(score), label

    async def run(self):
        await self.connect()
        await self.request_access()
        await self.refresh_and_find_headset()
        await self.connect_headset()
        await self.authorize()
        await self.create_session()
        await self.load_profile()
        await self.subscribe_com()

        print("\n🎯 Leyendo comandos mentales en tiempo real...")
        print(f"Mostrando solo score >= {MIN_SCORE:.2f}")
        print("Ctrl+C para salir.\n")

        while True:
            raw = await self.ws.recv()
            msg = json.loads(raw)

            if "warning" in msg:
                print(f"[warning] {msg['warning']}")
                continue

            if "com" not in msg:
                continue

            parsed = self.parse_com_sample(msg)
            if parsed is None:
                continue

            cmd, score, label = parsed

            if score >= MIN_SCORE:
                print(f"cmd={cmd:8s} score={score:.3f} -> {label}")


async def main():
    reader = CortexComReader(CLIENT_ID, CLIENT_SECRET, PROFILE_NAME)
    try:
        await reader.run()
    except KeyboardInterrupt:
        print("\n⛔ Fin por usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if reader.ws is not None:
            await reader.ws.close()
            print("Conexión cerrada.")


if __name__ == "__main__":
    asyncio.run(main())
