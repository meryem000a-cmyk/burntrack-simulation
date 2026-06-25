import asyncio
import json
import os
import sys
import jwt
from datetime import datetime, timedelta

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERREUR: Veuillez installer fastapi, uvicorn et pyjwt: pip install fastapi uvicorn pyjwt pydantic")
    sys.exit(1)

# Import the existing SimulationServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visualization.server import SimulationServer

app = FastAPI(title="BurnTrack API")

# CORS: credentials=True forbids wildcard origins per the CORS spec.
# Restrict to explicit origins from env (comma-separated), defaulting to localhost.
_cors_origins_env = os.environ.get("BURNTRACK_CORS_ORIGINS", "http://localhost:8765,http://127.0.0.1:8765")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT secret: read from env. If unset, generate a random per-process secret so
# tokens can never be forged by an attacker who reads this source file.
# Set BURNTRACK_JWT_SECRET explicitly for multi-process deployments (so tokens
# remain valid across reloads/workers).
SECRET_KEY = os.environ.get("BURNTRACK_JWT_SECRET") or os.urandom(32).hex()
# Admin credentials: read from env. If unset, the password is random per-process,
# which forces explicit configuration before any login can succeed in production.
ADMIN_USERNAME = os.environ.get("BURNTRACK_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("BURNTRACK_ADMIN_PASSWORD") or os.urandom(16).hex()
ALGORITHM = "HS256"

# Global simulation server instance
sim_server = SimulationServer()

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(req: LoginRequest):
    """Vérifie les identifiants et génère un jeton JWT."""
    if req.username == ADMIN_USERNAME and req.password == ADMIN_PASSWORD:
        expire = datetime.utcnow() + timedelta(hours=24)
        token = jwt.encode(
            {"sub": "admin", "exp": expire},
            SECRET_KEY, 
            algorithm=ALGORITHM
        )
        return {"token": token}
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

class WSWrap:
    """Wrapper pour rendre le WebSocket de FastAPI compatible avec l'ancien code."""
    def __init__(self, ws: WebSocket):
        self.ws = ws
    async def send(self, data: str):
        await self.ws.send_text(data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket. 
    Les commandes critiques (run_scenarios) nécessitent un token.
    """
    await websocket.accept()
    ws_wrapper = WSWrap(websocket)
    sim_server.ws_clients.add(ws_wrapper)
    print(f"[WS] Client connecté ({len(sim_server.ws_clients)} total)")

    # Send initial state (from server.py)
    from burntrack.engine.fuel_models import ALL_FUEL_MODELS
    await ws_wrapper.send(json.dumps({
        "type": "ready",
        "fuel_models": list(ALL_FUEL_MODELS.keys()),
        "grid": {
            "rows": sim_server.grid.rows if sim_server.grid else 50,
            "cols": sim_server.grid.cols if sim_server.grid else 50,
            "cell_size_m": sim_server.grid.cell_size if sim_server.grid else 30.0,
        } if sim_server.grid else None,
    }))

    is_admin = False

    try:
        while True:
            message = await websocket.receive_text()
            msg = json.loads(message)
            cmd = msg.get("cmd", "")

            # 1. Vérification d'authentification
            if cmd == "auth":
                token = msg.get("token", "")
                try:
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    if payload.get("sub") == "admin":
                        is_admin = True
                        await ws_wrapper.send(json.dumps({"type": "auth_success"}))
                except jwt.ExpiredSignatureError:
                    await ws_wrapper.send(json.dumps({"type": "error", "message": "Token expiré"}))
                except jwt.InvalidTokenError:
                    await ws_wrapper.send(json.dumps({"type": "error", "message": "Token invalide"}))
                continue

            # 2. Sécurisation des commandes administratives
            admin_commands = ["run_scenarios", "run_ensemble_scenario", "configure", "load_bouskoura"]
            if cmd in admin_commands and not is_admin:
                await ws_wrapper.send(json.dumps({
                    "type": "error", 
                    "message": f"Accès refusé. La commande '{cmd}' nécessite les droits administrateur."
                }))
                continue

            # 3. Traitement normal de la commande
            await sim_server._process_command(ws_wrapper, msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Erreur WS: {e}")
    finally:
        sim_server.ws_clients.discard(ws_wrapper)
        print(f"[WS] Client déconnecté ({len(sim_server.ws_clients)} total)")

if __name__ == "__main__":
    print("Démarrage de l'API BurnTrack (FastAPI)...")
    # Default to loopback for safety. Set BURNTRACK_HOST=0.0.0.0 to expose on the LAN.
    host = os.environ.get("BURNTRACK_HOST", "127.0.0.1")
    port = int(os.environ.get("BURNTRACK_PORT", "8765"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
