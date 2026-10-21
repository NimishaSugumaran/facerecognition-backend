"""
ArcFace Recognition System — Production Backend v2.0
FastAPI + InsightFace + OpenCV + MongoDB Atlas + WebSocket + JWT
"""

import os, cv2, base64, logging, asyncio, threading, time, uuid, json
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import deque
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# Suppress OpenCV MSMF verbose warnings on Windows (cap_msmf.cpp grabFrame errors)
# DSHOW backend is used instead, but suppress residual warnings
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")   # deprioritise MSMF
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")            # suppress WARN-level logs

# Arduino service — imported here so it is available before VideoWorker is built
from services.arduino_service import arduino_service, State as ArduinoState

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pymongo
from pymongo import MongoClient
from bson import ObjectId
import jwt as pyjwt

from insightface.app import FaceAnalysis

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — structured, rotating
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(
    "logs/arcface.log", maxBytes=5 * 1024 * 1024, backupCount=3,
    encoding="utf-8"          # always UTF-8 in the log file
)
file_handler.setFormatter(log_formatter)

# Force UTF-8 on Windows console (cp1252 cannot print many characters)
import sys
stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    if sys.platform == "win32" else sys.stdout
)
stream_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
logger = logging.getLogger("arcface")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — central, env-driven
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    MONGODB_URI: str         = os.getenv("MONGODB_URI", "")
    DB_NAME: str             = os.getenv("DB_NAME", "facerecog")
    JWT_SECRET: str          = os.getenv("JWT_SECRET", "arcface-super-secret-key-change-in-prod")
    JWT_ALGORITHM: str       = "HS256"
    JWT_EXPIRE_HOURS: int    = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
    ADMIN_USERNAME: str      = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str      = os.getenv("ADMIN_PASSWORD", "admin123")

    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    GUEST_START_HOUR: int    = int(os.getenv("GUEST_START_HOUR", "9"))
    GUEST_END_HOUR: int      = int(os.getenv("GUEST_END_HOUR", "18"))

    UNKNOWN_IMAGES_DIR: str  = os.getenv("UNKNOWN_IMAGES_DIR", "unknown_alerts")
    MOTION_THRESHOLD: int    = int(os.getenv("MOTION_THRESHOLD", "25"))
    MIN_MOTION_AREA: int     = int(os.getenv("MIN_MOTION_AREA", "500"))
    RESIZE_W: int            = int(os.getenv("RESIZE_W", "320"))
    RESIZE_H: int            = int(os.getenv("RESIZE_H", "240"))

    FRAME_SKIP: int          = int(os.getenv("FRAME_SKIP", "3"))
    WEBCAM_INDEX: int        = int(os.getenv("WEBCAM_INDEX", "0"))
    WS_FPS: float            = float(os.getenv("WS_FPS", "15"))

    # Cooldown in seconds after a successful grant — prevents repeated grants
    GRANT_COOLDOWN: float    = float(os.getenv("GRANT_COOLDOWN", "10"))

    BLINK_THRESHOLD: float   = float(os.getenv("BLINK_THRESHOLD", "0.25"))
    MOVEMENT_HISTORY: int    = int(os.getenv("MOVEMENT_HISTORY", "10"))

    # Arduino serial settings
    ARDUINO_PORT:     str    = os.getenv("ARDUINO_PORT",    "COM3")
    ARDUINO_BAUD:     int    = int(os.getenv("ARDUINO_BAUD", "9600"))
    ARDUINO_ENABLED:  bool   = os.getenv("ARDUINO_ENABLED", "true").lower() == "true"

cfg = Config()
os.makedirs(cfg.UNKNOWN_IMAGES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────────────────────────────────────
try:
    mongo_client = MongoClient(cfg.MONGODB_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()
    db         = mongo_client[cfg.DB_NAME]
    users_col  = db["users"]
    logs_col   = db["logs"]
    alerts_col = db["alerts"]

    users_col.create_index("user_id", unique=True)
    logs_col.create_index([("timestamp", pymongo.DESCENDING)])
    alerts_col.create_index([("timestamp", pymongo.DESCENDING)])
    logger.info("MongoDB connected OK")
except Exception as e:
    logger.error(f"MongoDB failed: {e}")
    raise RuntimeError(e)

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTFACE MODEL
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading ArcFace model ...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(320, 320))
logger.info("ArcFace model ready")

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY EMBEDDING CACHE
# ─────────────────────────────────────────────────────────────────────────────
embedding_cache: Dict[str, dict] = {}
cache_lock = threading.Lock()

def load_embeddings_to_cache():
    global embedding_cache
    with cache_lock:
        embedding_cache = {}
        for user in users_col.find():
            uid = user["user_id"]
            embedding_cache[uid] = {
                "name":       user["name"],
                "role":       user["role"],
                "embeddings": [np.array(e) for e in user["embeddings"]],
            }
    logger.info(f"Cache loaded - {len(embedding_cache)} users")

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-SPOOF STATE  (per-connection face tracking)
# ─────────────────────────────────────────────────────────────────────────────
class LivenessTracker:
    """
    Tracks face movement across frames to detect liveness.
    - EAR (Eye Aspect Ratio) approximated from landmark positions for blink detection
    - Centroid movement history for head-movement detection
    """
    def __init__(self, history: int = 10):
        self.centroids: deque = deque(maxlen=history)
        self.ear_history: deque = deque(maxlen=history)
        self.blink_detected = False
        self.movement_detected = False

    def update(self, face) -> None:
        # Face centroid from bounding box
        x1, y1, x2, y2 = face.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.centroids.append((cx, cy))

        # Rough EAR from 5-point landmarks if available
        if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5:
            # kps: [right_eye, left_eye, nose, right_mouth, left_mouth]
            re = face.kps[0]  # right eye
            le = face.kps[1]  # left eye
            ear = abs(re[1] - le[1]) / (abs(re[0] - le[0]) + 1e-6)
            self.ear_history.append(ear)
            if len(self.ear_history) >= 3:
                # Blink = sudden dip in EAR
                mid = list(self.ear_history)
                if mid[-2] < self.BLINK_THRESHOLD and mid[-1] > self.BLINK_THRESHOLD:
                    self.blink_detected = True

        # Head movement = centroid shift > 5 px
        if len(self.centroids) >= 3:
            xs = [c[0] for c in self.centroids]
            ys = [c[1] for c in self.centroids]
            span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            if span > 8:
                self.movement_detected = True

    BLINK_THRESHOLD = cfg.BLINK_THRESHOLD

    def is_live(self) -> bool:
        return self.blink_detected or self.movement_detected

    def reset(self):
        self.centroids.clear()
        self.ear_history.clear()
        self.blink_detected = False
        self.movement_detected = False

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND VIDEO WORKER
# ─────────────────────────────────────────────────────────────────────────────
class VideoWorker:
    """
    Runs in a background thread.
    Continuously reads webcam → motion detection → ArcFace recognition.
    Broadcasts events to all connected WebSocket clients.
    """
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.enabled = False             # Recognition toggle
        self.motion_enabled = True       # Motion-gate toggle
        self.cap: Optional[cv2.VideoCapture] = None

        self.latest_frame_b64: Optional[str] = None  # Latest JPEG for streaming
        self.latest_event: Optional[dict]     = None  # Latest recognition event
        self.fps_counter = 0
        self.fps_display = 0.0
        self._fps_time   = time.time()

        self._prev_gray: Optional[np.ndarray] = None
        self.liveness   = LivenessTracker()

        self._frame_index = 0           # for frame-skip
        self._lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None   # shared between capture+recog threads
        self._recog_thread: Optional[threading.Thread] = None

        # Cooldown — {user_id: last_granted_timestamp}
        # Prevents the same person being granted repeatedly every 1.5s.
        # After a grant, that user is suppressed for GRANT_COOLDOWN seconds.
        self._grant_cooldown: dict = {}
        self.GRANT_COOLDOWN = cfg.GRANT_COOLDOWN  # seconds, from .env

        # ── Arduino integration states ─────────────────────────────────────────
        # system_armed   : ultrasonic triggered — recognition should run
        # door_open      : OPEN command sent, servo is at 90°
        # cooldown_active: door was just opened, ignore all triggers
        # These mirror arduino_service's state for VideoWorker's own logic gate.
        self.system_armed    = False
        self.door_open       = False
        self.cooldown_active = False

    def _on_arduino_motion(self):
        """Callback: Arduino ultrasonic triggered → arm recognition."""
        self.system_armed    = True
        self.door_open       = False
        self.cooldown_active = False
        logger.info("VideoWorker armed by ultrasonic trigger")

    def _on_arduino_reset(self):
        """Callback: Arduino service reset to IDLE → disarm recognition."""
        self.system_armed    = False
        self.cooldown_active = False
        logger.info("VideoWorker disarmed by arduino_service reset")

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def start(self):
        """Start capture + recognition threads. Safe to call after stop()."""
        if self._thread and self._thread.is_alive():
            return   # already running
        # _stop_event was cleared in stop() so threads will run
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="frame-capture")
        self._recog_thread = threading.Thread(target=self._recognition_loop, daemon=True, name="recognition")
        self._thread.start()
        self._recog_thread.start()
        logger.info("Video worker started")

    def stop(self):
        """Stop both threads and release webcam. start() can be called again after this."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if hasattr(self, '_recog_thread') and self._recog_thread:
            self._recog_thread.join(timeout=3)
            self._recog_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._stop_event.clear()   # reset so start() works again
        self._prev_gray = None     # reset motion state
        logger.info("Video worker stopped — webcam released")

    # ── Capture loop — ONLY reads webcam and encodes JPEG, never blocks ───────
    def _capture_loop(self):
        # Small delay so browser has time to release the camera if it was using it
        time.sleep(0.5)

        # Use DSHOW on Windows — it releases the camera instantly unlike MSMF.
        # MSMF (-1072875772 error) holds a lock for 1-2s after release.
        backend = cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.WEBCAM_INDEX, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            logger.warning("Webcam not opened — retrying with default backend ...")
            time.sleep(1)
            self.cap.release()
            self.cap = cv2.VideoCapture(cfg.WEBCAM_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                logger.error("Could not open webcam — giving up")
                return

        logger.info(f"Webcam opened (backend: {'DSHOW' if backend == cv2.CAP_DSHOW else 'default'})")
        frame_delay = 1.0 / cfg.WS_FPS

        while not self._stop_event.is_set():
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # MSMF/DSHOW transient failure — skip this frame, don't spam log
                time.sleep(0.05)
                continue

            self._frame_index += 1

            # Encode and store latest frame — recognition thread reads this
            small = cv2.resize(frame, (cfg.RESIZE_W * 2, cfg.RESIZE_H * 2))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                self.latest_frame_b64  = base64.b64encode(buf).decode()
                self._latest_raw_frame = frame.copy()   # raw BGR for recognition

            # FPS counter
            self.fps_counter += 1
            elapsed = time.time() - self._fps_time
            if elapsed >= 1.0:
                self.fps_display = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self._fps_time   = time.time()

            # Sleep just enough to hit target FPS — never blocked by recognition
            time.sleep(max(0, frame_delay - (time.time() - t0)))

        self.cap.release()

    # ── Recognition loop — runs independently, never blocks capture ──────────
    def _recognition_loop(self):
        recog_interval = cfg.FRAME_SKIP / 30.0   # process every FRAME_SKIP frames worth of time

        while not self._stop_event.is_set():
            time.sleep(recog_interval)

            if not self.enabled:
                continue

            with self._lock:
                frame = getattr(self, '_latest_raw_frame', None)
            if frame is None:
                continue

            # Arduino gate
            arduino_mode = cfg.ARDUINO_ENABLED and arduino_service.connected
            if arduino_mode and not self.system_armed:
                continue

            # Motion gate
            motion = self._detect_motion(frame)
            if self.motion_enabled and not motion:
                _enqueue_event({"type": "no_motion", "motion_detected": False,
                                "fps": self.fps_display, "timestamp": _ts()})
                continue

            # Motion confirmed
            if arduino_mode and self.system_armed:
                arduino_service.set_recognizing()
                _enqueue_event({
                    "type":    "motion_confirmed",
                    "message": "OpenCV motion confirmed — running face recognition",
                    "timestamp": _ts(),
                })

            # Face recognition
            try:
                event = self._recognize(frame, motion)
                with self._lock:
                    self.latest_event = event
                _enqueue_event(event)

                # Arduino door control
                if arduino_mode:
                    if event.get("type") == "access_granted":
                        self.system_armed    = False
                        self.door_open       = True
                        self.cooldown_active = True
                        arduino_service.open_door()
                    elif event.get("type") in ("access_denied", "unknown_detected"):
                        self.system_armed = False
                        arduino_service.deny_and_reset()

            except Exception as e:
                logger.error(f"Recognition error: {e}")

    # ── Motion detection ─────────────────────────────────────────────────────
    def _detect_motion(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (cfg.RESIZE_W, cfg.RESIZE_H))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, thresh = cv2.threshold(diff, cfg.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) >= cfg.MIN_MOTION_AREA for c in cnts)

    # ── Recognition pipeline ─────────────────────────────────────────────────
    def _recognize(self, frame: np.ndarray, motion: bool) -> dict:
        rgb   = cv2.cvtColor(cv2.resize(frame, (cfg.RESIZE_W, cfg.RESIZE_H)), cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)

        if not faces:
            return {"type": "no_face", "motion_detected": motion,
                    "recognized": False, "fps": self.fps_display,
                    "timestamp": _ts()}

        # Largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        # Update liveness tracker
        self.liveness.update(face)
        is_live = self.liveness.is_live()

        # Bounding box (scale back to original)
        sx = frame.shape[1] / cfg.RESIZE_W
        sy = frame.shape[0] / cfg.RESIZE_H
        x1, y1, x2, y2 = face.bbox
        bbox = [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)]

        # Embedding
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        # Match
        with cache_lock:
            uid, name, role, sim = _find_best_match(emb)

        if uid is None:
            # Unknown
            threading.Thread(
                target=_save_unknown_alert, args=(frame.copy(), sim), daemon=True
            ).start()
            _log_access("Unknown", "denied", sim)
            self.liveness.reset()
            return {
                "type":             "unknown_detected",
                "motion_detected":  motion,
                "recognized":       False,
                "name":             "Unknown",
                "similarity":       round(sim, 4),
                "status":           "denied",
                "bbox":             bbox,
                "liveness":         is_live,
                "fps":              self.fps_display,
                "message":          "Unknown person — alert triggered",
                "timestamp":        _ts(),
            }

        # Access control
        if role == "admin":
            granted = True
            reason  = "Admin — 24/7 access"
        elif role == "guest":
            if cfg.GUEST_START_HOUR <= datetime.now().hour < cfg.GUEST_END_HOUR:
                granted = True
                reason  = f"Guest — access window {cfg.GUEST_START_HOUR}:00–{cfg.GUEST_END_HOUR}:00"
            else:
                granted = False
                reason  = f"Guest — outside access window"
        else:
            granted = False
            reason  = "Unknown role"

        ev_type = "access_granted" if granted else "access_denied"

        # ── Cooldown gate ─────────────────────────────────────────────────────
        # If this user was already granted within GRANT_COOLDOWN seconds,
        # return a silent "cooldown" event — no log, no door trigger.
        now = time.time()
        if granted and uid in self._grant_cooldown:
            elapsed = now - self._grant_cooldown[uid]
            if elapsed < self.GRANT_COOLDOWN:
                remaining = int(self.GRANT_COOLDOWN - elapsed)
                return {
                    "type":            "cooldown",
                    "motion_detected": motion,
                    "recognized":      True,
                    "user_id":         uid,
                    "name":            name,
                    "role":            role,
                    "similarity":      round(sim, 4),
                    "status":          "granted",
                    "bbox":            bbox,
                    "liveness":        is_live,
                    "fps":             self.fps_display,
                    "message":         f"Already granted — cooldown {remaining}s remaining",
                    "timestamp":       _ts(),
                }

        # Record grant time and log only on first recognition
        if granted:
            self._grant_cooldown[uid] = now

        _log_access(name, "granted" if granted else "denied", sim, role)
        logger.info(f"[{'GRANTED' if granted else 'DENIED'}] {name} ({role}) sim={sim:.3f} -> {ev_type}")

        return {
            "type":             ev_type,
            "motion_detected":  motion,
            "recognized":       True,
            "user_id":          uid,
            "name":             name,
            "role":             role,
            "similarity":       round(sim, 4),
            "status":           "granted" if granted else "denied",
            "bbox":             bbox,
            "liveness":         is_live,
            "fps":              self.fps_display,
            "message":          reason,
            "timestamp":        _ts(),
        }

# Singleton worker
video_worker = VideoWorker()

# ─────────────────────────────────────────────────────────────────────────────
# EVENT QUEUE  — thread-safe bridge from VideoWorker → main event loop
# ─────────────────────────────────────────────────────────────────────────────
import queue as _queue
_event_queue: _queue.Queue = _queue.Queue(maxsize=64)

def _enqueue_event(event: dict):
    """Called from the VideoWorker thread. Never awaited."""
    try:
        _event_queue.put_nowait(event)
    except _queue.Full:
        pass   # drop oldest if queue is full — prevents memory growth

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = threading.Lock()   # threading.Lock, NOT asyncio.Lock

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self.active.append(ws)
        logger.info(f"WS connected - {len(self.active)} clients")

    async def disconnect(self, ws: WebSocket):
        with self._lock:
            self.active = [c for c in self.active if c is not ws]
        logger.info(f"WS disconnected - {len(self.active)} clients")

    async def broadcast(self, data: dict):
        """Async broadcast — called from the main event loop only."""
        with self._lock:
            targets = list(self.active)   # snapshot under lock
        if not targets:
            return
        msg  = json.dumps(data)
        dead = []
        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

ws_manager = ConnectionManager()

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP — lifespan replaces deprecated on_event startup/shutdown
# ─────────────────────────────────────────────────────────────────────────────
# (app is defined after lifespan function below)

# ─── JWT helpers ──────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=cfg.JWT_EXPIRE_HOURS)
    return pyjwt.encode(
        {"sub": username, "exp": expire},
        cfg.JWT_SECRET, algorithm=cfg.JWT_ALGORITHM
    )

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    try:
        payload = pyjwt.decode(
            credentials.credentials, cfg.JWT_SECRET,
            algorithms=[cfg.JWT_ALGORITHM]
        )
        return payload["sub"]
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

# ─── Shared helpers ────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _find_best_match(query_emb: np.ndarray):
    best_sim, best_uid, best_name, best_role = -1.0, None, None, None
    for uid, data in embedding_cache.items():
        sims    = [float(np.dot(query_emb, e)) for e in data["embeddings"]]
        avg_sim = float(np.mean(sims))
        if avg_sim > best_sim:
            best_sim, best_uid, best_name, best_role = avg_sim, uid, data["name"], data["role"]
    if best_sim >= cfg.SIMILARITY_THRESHOLD:
        return best_uid, best_name, best_role, best_sim
    return None, None, None, best_sim

def _log_access(name, status, similarity, role=""):
    logs_col.insert_one({
        "name": name, "role": role, "status": status,
        "similarity": round(similarity, 4),
        "timestamp":  datetime.utcnow()
    })

def _save_unknown_alert(frame: np.ndarray, similarity: float):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"{cfg.UNKNOWN_IMAGES_DIR}/unknown_{ts}.jpg"
    cv2.imwrite(path, frame)
    alerts_col.insert_one({
        "timestamp":      datetime.utcnow(),
        "image_path":     path,
        "best_similarity": similarity,
        "message":        "Unknown person attempted access",
    })
    logger.warning(f"ALERT - Unknown person saved: {path}")

def _b64_to_frame(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame

def _get_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = largest.embedding
    return emb / np.linalg.norm(emb)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────
    # Run MongoDB cache load in thread so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_embeddings_to_cache)

    # Queue-drainer: picks up events from VideoWorker thread and broadcasts via WS
    drainer_task = asyncio.create_task(_queue_drainer())

    # Wire Arduino callbacks and start service
    arduino_service.set_callbacks(
        on_motion  = video_worker._on_arduino_motion,
        on_reset   = video_worker._on_arduino_reset,
        enqueue_fn = _enqueue_event,
    )
    if cfg.ARDUINO_ENABLED:
        arduino_service.start()
    else:
        logger.info("ARDUINO_ENABLED=false — Arduino integration skipped")

    logger.info("ArcFace API v2.0 ready")
    yield
    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    drainer_task.cancel()
    video_worker.stop()
    arduino_service.stop()

app = FastAPI(title="ArcFace Production API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _queue_drainer():
    """
    Drains _event_queue on the main asyncio event loop.
    The VideoWorker thread posts events via _enqueue_event() (thread-safe).
    This coroutine picks them up and does the actual WebSocket broadcast,
    which MUST happen on the event loop that owns the WS connections.
    """
    while True:
        try:
            # Non-blocking check every 20ms
            event = _event_queue.get_nowait()
            await ws_manager.broadcast(event)
        except _queue.Empty:
            await asyncio.sleep(0.02)
        except Exception as e:
            logger.error(f"Queue drainer error: {e}")
            await asyncio.sleep(0.1)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    name:   str
    role:   str
    images: List[str]

class RecognizeRequest(BaseModel):
    image:        str
    check_motion: bool = True

class DeleteUserRequest(BaseModel):
    user_id: str

class WorkerControlRequest(BaseModel):
    enabled:        Optional[bool] = None
    motion_enabled: Optional[bool] = None

# ─────────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def login(req: LoginRequest):
    if req.username != cfg.ADMIN_USERNAME or req.password != cfg.ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid credentials")
    token = create_token(req.username)
    logger.info(f"Login: {req.username}")
    return {"access_token": token, "token_type": "bearer", "expires_in": cfg.JWT_EXPIRE_HOURS * 3600}

@app.get("/auth/me")
async def me(user: str = Depends(verify_token)):
    return {"username": user}

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET  /ws/live
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws_manager.connect(ws)
    # Send system status immediately on connect
    await ws.send_text(json.dumps({
        "type":             "system_status",
        "worker_enabled":   video_worker.enabled,
        "motion_enabled":   video_worker.motion_enabled,
        "cached_users":     len(embedding_cache),
        "fps":              video_worker.fps_display,
        "timestamp":        _ts(),
    }))
    try:
        # Stream latest frame every ~67ms (≈15 fps) in parallel
        async def stream_frames():
            interval = 1.0 / cfg.WS_FPS
            while True:
                frame_b64 = video_worker.latest_frame_b64
                if frame_b64:
                    try:
                        await ws.send_text(json.dumps({
                            "type":  "frame",
                            "data":  frame_b64,
                            "fps":   video_worker.fps_display,
                            "timestamp": _ts(),
                        }))
                    except Exception:
                        break
                await asyncio.sleep(interval)

        frame_task = asyncio.create_task(stream_frames())

        # Listen for control messages from the client
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "toggle_recognition":
                    new_val = bool(msg.get("value", False))
                    video_worker.enabled = new_val
                    if new_val:
                        # Start worker — restarts cleanly after a pause
                        if not (video_worker._thread and video_worker._thread.is_alive()):
                            # Run start in executor — avoids any blocking on event loop
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, video_worker.start)
                        logger.info("Recognition enabled via WebSocket")
                    else:
                        # PAUSE — stop in executor so thread joins don't block event loop
                        # (blocking join inside async handler freezes WS keepalives → Stream Offline)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, video_worker.stop)
                        with video_worker._lock:
                            video_worker.latest_frame_b64  = None
                            video_worker._latest_raw_frame = None
                        logger.info("Recognition paused — webcam released for browser")
                    await ws.send_text(json.dumps({
                        "type":                "control_ack",
                        "recognition_enabled": video_worker.enabled,
                        "worker_running":      video_worker._thread is not None and video_worker._thread.is_alive(),
                        "timestamp":           _ts(),
                    }))
                elif cmd == "toggle_motion":
                    video_worker.motion_enabled = bool(msg.get("value", True))
                    await ws.send_text(json.dumps({
                        "type": "control_ack",
                        "motion_enabled": video_worker.motion_enabled,
                        "timestamp": _ts(),
                    }))
                elif cmd == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "timestamp": _ts()}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        frame_task.cancel()
        await ws_manager.disconnect(ws)

# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "ArcFace Production API", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "cached_users":    len(embedding_cache),
        "worker_running":  video_worker._thread is not None and video_worker._thread.is_alive(),
        "worker_enabled":  video_worker.enabled,
        "ws_clients":      len(ws_manager.active),
        "fps":             video_worker.fps_display,
        "db_connected":    True,
        "arduino":         arduino_service.get_status(),
    }

# ── Register (JWT-protected) ──────────────────────────────────────────────────
def _extract_embeddings_sync(images: List[str]):
    """
    CPU-heavy embedding extraction — runs in a thread pool executor
    so it does NOT block the async event loop (which caused the timeout).
    """
    embeddings, failed = [], 0
    for b64 in images:
        try:
            frame = _b64_to_frame(b64)
            small = cv2.resize(frame, (cfg.RESIZE_W, cfg.RESIZE_H))
            emb   = _get_embedding(small)
            if emb is not None:
                embeddings.append(emb.tolist())
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"Frame error: {e}")
            failed += 1
    return embeddings, failed

@app.post("/register")
async def register(req: RegisterRequest):
    """Register a new user — no JWT required (local network system)."""
    if len(req.images) < 5:
        raise HTTPException(400, "Provide >= 5 images")
    if req.role not in ("admin", "guest"):
        raise HTTPException(400, "Role must be 'admin' or 'guest'")

    # Run CPU-heavy work in a thread so the event loop stays alive
    # (prevents the 30 000 ms timeout the frontend was seeing)
    loop = asyncio.get_event_loop()
    embeddings, failed = await loop.run_in_executor(
        None, _extract_embeddings_sync, req.images
    )

    if len(embeddings) < 3:
        raise HTTPException(400, f"Only {len(embeddings)} valid faces extracted ({failed} failed). Ensure your face is clearly visible and well lit.")

    user_id  = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}_{req.name.lower().replace(' ', '_')}"
    user_doc = {
        "user_id":     user_id,
        "name":        req.name,
        "role":        req.role,
        "embeddings":  embeddings,
        "created_at":  datetime.utcnow(),
        "frame_count": len(embeddings),
    }
    try:
        users_col.insert_one(user_doc)
    except pymongo.errors.DuplicateKeyError:
        raise HTTPException(409, "ID conflict, retry")

    with cache_lock:
        embedding_cache[user_id] = {
            "name":       req.name,
            "role":       req.role,
            "embeddings": [np.array(e) for e in embeddings],
        }

    logger.info(f"Registered {req.name} ({user_id}) - {len(embeddings)} embeddings")

    await ws_manager.broadcast({
        "type":      "user_registered",
        "name":      req.name,
        "role":      req.role,
        "user_id":   user_id,
        "timestamp": _ts(),
    })

    return {
        "success":           True,
        "user_id":           user_id,
        "name":              req.name,
        "role":              req.role,
        "embeddings_stored": len(embeddings),
        "frames_failed":     failed,
    }

# ── Single-frame recognize (fallback for non-WS clients) ─────────────────────
@app.post("/recognize")
async def recognize(req: RecognizeRequest, background_tasks: BackgroundTasks):
    try:
        frame = _b64_to_frame(req.image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    motion = video_worker._detect_motion(frame)
    if req.check_motion and not motion:
        return {"motion_detected": False, "recognized": False,
                "status": "no_motion", "message": "No motion detected"}

    emb = _get_embedding(cv2.resize(frame, (cfg.RESIZE_W, cfg.RESIZE_H)))
    if emb is None:
        return {"motion_detected": motion, "recognized": False,
                "status": "no_face",  "message": "No face in frame"}

    with cache_lock:
        uid, name, role, sim = _find_best_match(emb)

    if uid is None:
        background_tasks.add_task(_save_unknown_alert, frame.copy(), sim)
        _log_access("Unknown", "denied", sim)
        return {"motion_detected": motion, "recognized": False, "name": "Unknown",
                "similarity": round(sim, 4), "status": "denied",
                "message": "Unknown person — alert triggered"}

    if role == "admin":
        granted, reason = True, "Admin access"
    elif role == "guest":
        if cfg.GUEST_START_HOUR <= datetime.now().hour < cfg.GUEST_END_HOUR:
            granted, reason = True, f"Guest — within window"
        else:
            granted, reason = False, "Guest — outside window"
    else:
        granted, reason = False, "Unknown role"

    status_str = "granted" if granted else "denied"
    _log_access(name, status_str, sim, role)

    return {
        "motion_detected": motion, "recognized": True,
        "user_id": uid, "name": name, "role": role,
        "similarity": round(sim, 4), "status": status_str,
        "message": reason, "timestamp": _ts()
    }

# ── Worker control (JWT-protected) ───────────────────────────────────────────
@app.post("/worker/control")
async def control_worker(req: WorkerControlRequest, user: str = Depends(verify_token)):
    if req.enabled is not None:
        video_worker.enabled = req.enabled
        if req.enabled and not (video_worker._thread and video_worker._thread.is_alive()):
            video_worker.start()
    if req.motion_enabled is not None:
        video_worker.motion_enabled = req.motion_enabled
    await ws_manager.broadcast({
        "type":               "system_status",
        "worker_enabled":     video_worker.enabled,
        "motion_enabled":     video_worker.motion_enabled,
        "cached_users":       len(embedding_cache),
        "fps":                video_worker.fps_display,
        "timestamp":          _ts(),
    })
    return {
        "worker_enabled":   video_worker.enabled,
        "motion_enabled":   video_worker.motion_enabled,
        "worker_running":   video_worker._thread is not None and video_worker._thread.is_alive(),
    }

@app.get("/worker/status")
async def worker_status():
    return {
        "enabled":  video_worker.enabled,
        "motion_enabled": video_worker.motion_enabled,
        "running":  video_worker._thread is not None and video_worker._thread.is_alive(),
        "fps":      video_worker.fps_display,
        "ws_clients": len(ws_manager.active),
    }

@app.get("/arduino/status")
async def arduino_status():
    """Live state of the Arduino service and door state machine."""
    return arduino_service.get_status()

@app.post("/arduino/simulate-trigger")
async def arduino_simulate_trigger():
    """
    Dev/test endpoint — simulate an ultrasonic trigger without real hardware.
    Useful for testing the full pipeline on a laptop.
    """
    video_worker._on_arduino_motion()
    _enqueue_event({
        "type":      "ultrasonic_triggered",
        "message":   "SIMULATED ultrasonic trigger (no hardware)",
        "timestamp": _ts(),
    })
    return {"simulated": True, "state": arduino_service.get_status()}

# ── Logs ─────────────────────────────────────────────────────────────────────
@app.get("/logs")
async def get_logs(limit: int = 50, status: Optional[str] = None, name: Optional[str] = None):
    q = {}
    if status: q["status"] = status
    if name:   q["name"]   = {"$regex": name, "$options": "i"}
    docs = []
    for doc in logs_col.find(q).sort("timestamp", pymongo.DESCENDING).limit(limit):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        docs.append(doc)
    return {"logs": docs, "count": len(docs)}

# ── Users ─────────────────────────────────────────────────────────────────────
@app.get("/users")
async def get_users():
    docs = []
    for doc in users_col.find({}, {"embeddings": 0}):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        docs.append(doc)
    return {"users": docs, "count": len(docs)}

# ── Delete User (JWT-protected) ───────────────────────────────────────────────
@app.delete("/user")
async def delete_user(req: DeleteUserRequest, user: str = Depends(verify_token)):
    result = users_col.delete_one({"user_id": req.user_id})
    if result.deleted_count == 0:
        raise HTTPException(404, f"User {req.user_id} not found")
    with cache_lock:
        embedding_cache.pop(req.user_id, None)
    logger.info(f"Deleted user: {req.user_id}")
    await ws_manager.broadcast({"type": "user_deleted", "user_id": req.user_id, "timestamp": _ts()})
    return {"success": True, "deleted_user_id": req.user_id}

# ── Alerts ────────────────────────────────────────────────────────────────────
@app.get("/alerts")
async def get_alerts(limit: int = 20):
    docs = []
    for doc in alerts_col.find().sort("timestamp", pymongo.DESCENDING).limit(limit):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        docs.append(doc)
    return {"alerts": docs, "count": len(docs)}

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
async def get_stats():
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "total_users":    users_col.count_documents({}),
        "total_logs":     logs_col.count_documents({}),
        "granted":        logs_col.count_documents({"status": "granted"}),
        "denied":         logs_col.count_documents({"status": "denied"}),
        "unknown_alerts": alerts_col.count_documents({}),
        "today_logs":     logs_col.count_documents({"timestamp": {"$gte": today}}),
        "cached_users":   len(embedding_cache),
        "ws_clients":     len(ws_manager.active),
        "fps":            video_worker.fps_display,
        "worker_enabled": video_worker.enabled,
    }

# ── Reload cache ──────────────────────────────────────────────────────────────
@app.post("/reload-cache")
async def reload_cache():
    load_embeddings_to_cache()
    return {"success": True, "cached_users": len(embedding_cache)}

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)