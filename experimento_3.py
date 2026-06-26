import asyncio
import json
import ssl
import csv
import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch
import websockets

# =========================================================
# CONFIGURACIÓN
# =========================================================
CLIENT_ID     = # tu id de cliente
CLIENT_SECRET = # tu client secret
URL           = "wss://localhost:6868"

PROFILE_NAME  = "CASO DIF"

MIN_SCORE     = 0.40  # aumentar --> mejor calidad de datos
SAMPLE_SEC    = 3.0
FS            = 128
MS_PER_SAMPLE = 1000 / FS
BUFFER_SIZE   = int(SAMPLE_SEC * FS)   # 384 muestras
COOLDOWN_SEC  = 4.0
MAX_SAMPLES_PER_CLASS = 40
OUTPUT_DIR    = "experimento_3"

# Canales: array EEG Cortex → [sample, ?, AF3, T7, Pz, T8, AF4, marker]
CHANNEL_NAMES   = ["AF3", "T7", "Pz", "T8", "AF4"]
CHANNEL_INDICES = [2, 3, 4, 5, 6]

LABEL_MAP = {
   
    "lift":     "pelota botando",
    "push":     "pestañear",
    "neutral":  "relax",
}


# =========================================================
# BANDAS A CALCULAR POR CANAL
# Formato: (nombre_columna, freq_min, freq_max)
# =========================================================
BANDS_PER_CHANNEL = {
    "AF3": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
    "T7":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "Pz":  [("alpha", 8, 12), ("beta",  13, 30)],
    "T8":  [("theta", 4, 7),  ("mu",    8, 12), ("beta", 13, 30)],
    "AF4": [("theta", 4, 7),  ("alpha", 8, 12), ("beta", 13, 30)],
}
 
# Columnas de bandas que se añaden al CSV (en orden)
# Se generan automáticamente a partir de BANDS_PER_CHANNEL
BAND_COLS = []
for ch in CHANNEL_NAMES:
    for band_name, _, _ in BANDS_PER_CHANNEL[ch]:
        BAND_COLS.append(f"{ch}_{band_name}")
 
# Cabecera completa del CSV:
# timestamp | 5 canales crudos | 14 columnas de bandas
CSV_HEADER = ["timestamp"] + CHANNEL_NAMES + BAND_COLS
 
 
# =========================================================
# FUNCIONES DE PROCESADO DE SEÑAL
# =========================================================
def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    fs: int = FS, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low  = lowcut  / nyquist
    high = highcut / nyquist
    # Clamp para evitar valores fuera de rango
    low  = max(1e-4, min(low,  0.9999))
    high = max(1e-4, min(high, 0.9999))
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
 
 
def bandpower(signal: np.ndarray, fmin: float, fmax: float,
              fs: int = FS) -> float:
    """Potencia relativa de una banda usando Welch."""
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
 
    idx_band  = (freqs >= fmin) & (freqs <= fmax)
    idx_total = (freqs >= 1)    & (freqs <= 40)
 
    power_band  = np.trapezoid(psd[idx_band],  freqs[idx_band])
    power_total = np.trapezoid(psd[idx_total], freqs[idx_total])
 
    if power_total <= 0:
        return 0.0
    return float(power_band / power_total)
 
 
def compute_band_features(buffer: deque) -> Dict[str, float]:
    """
    Recibe el buffer circular (BUFFER_SIZE × 5 canales) y devuelve
    un dict con la potencia relativa de cada banda por canal.
    """
    # Convertir buffer a array numpy (384 × 5)
    arr = np.array(list(buffer), dtype=float)  # shape (384, 5)
 
    features = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = arr[:, ch_idx]
        signal = signal - np.mean(signal)  # eliminar offset DC
 
        # Filtrado general 1-40 Hz antes de calcular bandas
        try:
            signal = bandpass_filter(signal, 1.0, 40.0)
        except Exception:
            pass  # si falla el filtro, usar señal sin filtrar
 
        for band_name, fmin, fmax in BANDS_PER_CHANNEL[ch_name]:
            col_name = f"{ch_name}_{band_name}"
            features[col_name] = bandpower(signal, fmin, fmax)
 
    return features
 
 
# =========================================================
# CLASE PRINCIPAL
# =========================================================
class CortexCapture:
    def __init__(self):
        self.ws = None
        self.req_id = 1
 
        self.cortex_token: Optional[str] = None
        self.headset_id:   Optional[str] = None
        self.session_id:   Optional[str] = None
        self.com_cols      = []
 
        self.eeg_buffer = deque(maxlen=BUFFER_SIZE)
        self.last_saved: Dict[str, float] = {}
        self.saved_count: Dict[str, int]  = {}
 
        os.makedirs(OUTPUT_DIR, exist_ok=True)
 
        # Contar ficheros ya existentes para no sobreescribir
        for label in LABEL_MAP.values():
            safe_label = label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            existing = [
                f for f in os.listdir(OUTPUT_DIR)
                if f.startswith(safe_label + ".") and f.endswith(".csv")
            ]
            self.saved_count[label] = len(existing)
            if existing:
                print(f"  Ya existen {len(existing)} muestras de '{label}', se continúa desde {len(existing)+1}")
 
    # ----------------------------------------------------------
    # CORTEX API
    # ----------------------------------------------------------
    async def rpc(self, method: str, params: Dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id":      self.req_id,
            "method":  method,
            "params":  params,
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
 
    async def setup(self):
        ssl_ctx = ssl._create_unverified_context()
        self.ws  = await websockets.connect(URL, ssl=ssl_ctx)
        print("Conectado a Cortex")
 
        result = await self.rpc("requestAccess", {
            "clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET
        })
        if not result.get("accessGranted", False):
            raise RuntimeError("Acceso denegado. Aprueba la app en Emotiv Launcher.")
        print("Acceso concedido")
 
        try:
            await self.rpc("controlDevice", {"command": "refresh"})
            await asyncio.sleep(2)
        except Exception:
            pass
 
        headsets = await self.rpc("queryHeadsets", {})
        if not headsets:
            raise RuntimeError("No se detectó ningún headset.")
        self.headset_id = headsets[0]["id"]
        print(f"Headset: {self.headset_id}")
 
        await self.rpc("controlDevice", {
            "command": "connect", "headset": self.headset_id
        })
        await asyncio.sleep(3)
        print("Headset conectado")
 
        result = await self.rpc("authorize", {
            "clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET, "debit": 1
        })
        self.cortex_token = result["cortexToken"]
        print("Autorización correcta")
 
        result = await self.rpc("createSession", {
            "cortexToken": self.cortex_token,
            "headset":     self.headset_id,
            "status":      "active",
        })
        self.session_id = result["id"]
        print(f"Sesión: {self.session_id}")
 
        result = await self.rpc("setupProfile", {
            "cortexToken": self.cortex_token,
            "headset":     self.headset_id,
            "profile":     PROFILE_NAME,
            "status":      "load",
        })
        print(f"Perfil cargado: {result.get('name', PROFILE_NAME)}")
 
        result = await self.rpc("subscribe", {
            "cortexToken": self.cortex_token,
            "session":     self.session_id,
            "streams":     ["eeg", "com"],
        })
        for stream in result.get("success", []):
            if stream.get("streamName") == "com":
                self.com_cols = stream.get("cols", ["act", "pow"])
        print(f"Suscrito a EEG + COM  |  columnas COM: {self.com_cols}")
 
        # Mostrar cabecera del CSV para que el usuario sepa qué se va a guardar
        print(f"\n  Columnas CSV ({len(CSV_HEADER)}):")
        print(f"  {CSV_HEADER}\n")
 
    # ----------------------------------------------------------
    # GUARDAR CSV
    # ----------------------------------------------------------
    def save_sample(self, label: str, score: float) -> bool:
        now = asyncio.get_event_loop().time()
 
        # Cooldown
        if label in self.last_saved:
            if now - self.last_saved[label] < COOLDOWN_SEC:
                return False
 
        # Máximo por clase
        count = self.saved_count.get(label, 0)
        if count >= MAX_SAMPLES_PER_CLASS:
            return False
 
        # Buffer lleno
        if len(self.eeg_buffer) < BUFFER_SIZE:
            return False
 
        # --- Calcular bandas sobre el buffer completo ---
        band_features = compute_band_features(self.eeg_buffer)
 
        # Nombre fichero: label.NN.csv
        idx      = count + 1
 
        # Limpiar el label para que sea un nombre de fichero válido
        safe_label = label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        filename = f"{safe_label}.{idx:02d}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
 
        # Convertir buffer a array para escribir señal cruda
        raw_arr = np.array(list(self.eeg_buffer), dtype=float)  # (384, 5)
 
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
 
            for i in range(len(raw_arr)):
                timestamp = i * MS_PER_SAMPLE
                raw_vals  = raw_arr[i].tolist()
 
                # Las bandas son iguales para toda la muestra (calculadas sobre
                # los 3s completos) → se repiten en cada fila para que Edge
                # Impulse pueda usarlas como features adicionales por ventana
                band_vals = [band_features[col] for col in BAND_COLS]
 
                writer.writerow([timestamp] + raw_vals + band_vals)
 
        self.last_saved[label]  = now
        self.saved_count[label] = idx
        return True
 
    # ----------------------------------------------------------
    # LOOP PRINCIPAL
    # ----------------------------------------------------------
    async def run(self):
        await self.setup()
 
        print("=" * 60)
        print("  CAPTURA ACTIVA")
        print(f"  Umbral confianza : {MIN_SCORE:.2f}")
        print(f"  Duración muestra : {SAMPLE_SEC} s  ({BUFFER_SIZE} muestras)")
        print(f"  Cooldown         : {COOLDOWN_SEC} s entre guardados")
        print(f"  Máx. por clase   : {MAX_SAMPLES_PER_CLASS}")
        print(f"  Señal cruda      : {len(CHANNEL_NAMES)} canales (AF3, T7, Pz, T8, AF4)")
        print(f"  Bandas           : {len(BAND_COLS)} columnas → {BAND_COLS}")
        print(f"  Carpeta salida   : {OUTPUT_DIR}/")
        print("  Ctrl+C para salir")
        print("=" * 60 + "\n")
 
        async for raw in self.ws:
            msg = json.loads(raw)
 
            if "warning" in msg:
                print(f"  [warning] {msg['warning']}")
                continue
 
            # --- EEG: alimentar buffer ---
            if "eeg" in msg:
                eeg = msg["eeg"]
                sample = [eeg[idx] if len(eeg) > idx else 0.0
                          for idx in CHANNEL_INDICES]
                self.eeg_buffer.append(sample)
                continue
 
            # --- COM: detectar estado mental ---
            if "com" in msg:
                values = msg["com"]
                if not values or not self.com_cols:
                    continue
 
                data  = dict(zip(self.com_cols, values))
                cmd   = data.get("act", "")
                score = float(data.get("pow", 0.0) or 0.0)
                print(f"\nCOMANDO RECIBIDO: '{cmd}' | score={score:.2f}")
                label = LABEL_MAP.get(cmd, None)
 
                if label is None:
                    print(f"No está en LABEL_MAP: '{cmd}'")
                    continue
 
                bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                count = self.saved_count.get(label, 0)
                print(f"  {cmd:14s} [{bar}] {score:.2f}  →  {label}  [{count}/{MAX_SAMPLES_PER_CLASS}]",
                      end="\r", flush=True)
 
                if score >= MIN_SCORE:
                    saved = self.save_sample(label, score)
                    if saved:
                        count = self.saved_count.get(label, 0)
                        print(f"\n  GUARDADO: {label}.{count:02d}.csv  "
                              f"(score={score:.2f})  [{count}/{MAX_SAMPLES_PER_CLASS}]")
 
                        # ¿Todas las clases completas?
                        clases_completas = [
                            l for l, c in self.saved_count.items()
                            if c >= MAX_SAMPLES_PER_CLASS
                        ]
                        if len(clases_completas) == len(LABEL_MAP):
                            print("\n ¡Todas las clases completadas!")
                            break
 
        # Resumen final
        print("\n\n" + "=" * 60)
        print("  RESUMEN DE CAPTURA")
        print("=" * 60)
        for label, count in self.saved_count.items():
            print(f"  {label:30s} : {count:3d} muestras")
        total = sum(self.saved_count.values())
        print(f"  {'TOTAL':30s} : {total:3d} muestras")
        print(f"\n  Carpeta: {os.path.abspath(OUTPUT_DIR)}/")
        print("=" * 60)
        print("\n  PRÓXIMOS PASOS EN EDGE IMPULSE:")
        print("  1. Ve a tu proyecto → Data acquisition")
        print("  2. Haz clic en 'Upload existing data'")
        print(f"  3. Sube todos los CSV de la carpeta '{OUTPUT_DIR}/'")
        print("  4. El label se asigna automáticamente por el nombre del fichero")
        print("=" * 60)
 
 
# =========================================================
# MAIN
# =========================================================
async def main():
    capture = CortexCapture()
    try:
        await capture.run()
    except KeyboardInterrupt:
        print("\n\n Captura detenida por el usuario.")
        if capture.saved_count:
            print("Muestras guardadas hasta ahora:")
            for label, count in capture.saved_count.items():
                print(f"  {label}: {count}")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        if capture.ws:
            await capture.ws.close()
            print("Conexión cerrada.")
 
 
if __name__ == "__main__":
    asyncio.run(main())
