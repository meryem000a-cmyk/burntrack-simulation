import asyncio
import json
import os
import sys
import hashlib
import jwt
from datetime import datetime, timedelta
from fastapi import Header

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "burntrack_super_secret_key"
ALGORITHM = "HS256"
ADMINS_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "admins.json"))

# Global simulation server instance
sim_server = SimulationServer()

def hash_password(password: str, salt: str = "burntrack_salt") -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()

def load_admins() -> dict:
    if not os.path.exists(ADMINS_FILE):
        os.makedirs(os.path.dirname(ADMINS_FILE), exist_ok=True)
        default_admins = {"admin": hash_password("admin")}
        with open(ADMINS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_admins, f, indent=4)
        return default_admins
    try:
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"admin": hash_password("admin")}

def save_admins(admins: dict):
    os.makedirs(os.path.dirname(ADMINS_FILE), exist_ok=True)
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(admins, f, indent=4)

def get_current_admin(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Jeton d'autorisation manquant")
    try:
        parts = authorization.split(" ")
        token = parts[1] if len(parts) > 1 else parts[0]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        admins = load_admins()
        if username in admins:
            return username
    except Exception:
        pass
    raise HTTPException(status_code=401, detail="Jeton d'accès invalide ou expiré")

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(req: LoginRequest):
    """Vérifie les identifiants et génère un jeton JWT."""
    admins = load_admins()
    hashed = hash_password(req.password)
    if req.username in admins and admins[req.username] == hashed:
        expire = datetime.utcnow() + timedelta(hours=24)
        token = jwt.encode(
            {"sub": req.username, "exp": expire},
            SECRET_KEY, 
            algorithm=ALGORITHM
        )
        return {"token": token}
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

# --- ENDPOINTS POUR LA GESTION DES ADMINS ---
@app.get("/api/admins")
def get_admins(current_user: str = Depends(get_current_admin)):
    """Retourne la liste des administrateurs."""
    return list(load_admins().keys())

@app.post("/api/admins")
def add_admin(req: LoginRequest, current_user: str = Depends(get_current_admin)):
    """Ajoute ou modifie un administrateur."""
    admins = load_admins()
    admins[req.username] = hash_password(req.password)
    save_admins(admins)
    return {"status": "ok"}

@app.delete("/api/admins/{username}")
def delete_admin(username: str, current_user: str = Depends(get_current_admin)):
    """Supprime un administrateur."""
    if username == "admin":
        raise HTTPException(status_code=400, detail="Impossible de supprimer l'administrateur principal 'admin'")
    admins = load_admins()
    if username not in admins:
        raise HTTPException(status_code=404, detail="Administrateur non trouvé")
    del admins[username]
    save_admins(admins)
    return {"status": "ok"}

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
                    sub = payload.get("sub")
                    if sub in load_admins():
                        is_admin = True
                        await ws_wrapper.send(json.dumps({"type": "auth_success"}))
                except jwt.ExpiredSignatureError:
                    await ws_wrapper.send(json.dumps({"type": "error", "message": "Token expiré"}))
                except jwt.InvalidTokenError:
                    await ws_wrapper.send(json.dumps({"type": "error", "message": "Token invalide"}))
                continue

            # 2. Sécurisation des commandes administratives
            admin_commands = ["run_scenarios", "run_ensemble_scenario", "configure", "load_bouskoura", "play", "pause", "step", "reset", "ignite"]
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
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
