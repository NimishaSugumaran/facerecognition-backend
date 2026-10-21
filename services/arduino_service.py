"""
arduino_service.py
Hardware bridge between Arduino (HC-SR04 + Servo) and the ArcFace backend.

State machine:
  IDLE → ARMED (ultrasonic trigger) → RECOGNIZING → DOOR_OPEN → COOLDOWN → IDLE

Thread model:
  • _reader_thread  : reads serial lines from Arduino (daemon)
  • _timer_thread   : fires timeout / cooldown resets (daemon)
  • All state mutations are protected by a threading.Lock
  • Events are pushed into the shared _event_queue (same one used by VideoWorker)
    so the async queue-drainer broadcasts them to WebSocket clients automatically.
"""

import os
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger("arcface.arduino")

# ── Timing constants (all overridable via .env) ───────────────────────────────
SERIAL_PORT      = os.getenv("ARDUINO_PORT",    "COM3")      # Windows: COM3, Linux: /dev/ttyUSB0
BAUD_RATE        = int(os.getenv("ARDUINO_BAUD", "9600"))
DEBOUNCE_SEC     = float(os.getenv("ARDUINO_DEBOUNCE",   "2"))    # ignore repeat triggers within 2s
TRIGGER_TIMEOUT  = float(os.getenv("ARDUINO_TIMEOUT",    "15"))   # disarm after 15s if no face found (CPU inference needs time)
DOOR_OPEN_SEC    = float(os.getenv("DOOR_OPEN_SEC",      "10"))   # door stays open 10s
COOLDOWN_SEC     = float(os.getenv("COOLDOWN_SEC",        "10"))   # ignore all triggers for 10s after open

# ── System states ─────────────────────────────────────────────────────────────
class State:
    IDLE         = "IDLE"
    ARMED        = "ARMED"        # ultrasonic triggered, waiting for face
    RECOGNIZING  = "RECOGNIZING"  # face recognition pipeline active
    DOOR_OPEN    = "DOOR_OPEN"    # OPEN sent, door is open
    COOLDOWN     = "COOLDOWN"     # door closed, ignoring all triggers


class ArduinoService:
    """
    Manages the full lifecycle:
      Serial read → state transitions → serial write → timer resets

    Usage:
        from services.arduino_service import arduino_service
        arduino_service.start()
        ...
        arduino_service.open_door()   # called by VideoWorker on access_granted
        arduino_service.stop()
    """

    def __init__(self):
        self._ser                  = None          # serial.Serial instance
        self._lock                 = threading.Lock()
        self._state                = State.IDLE
        self._last_trigger_time    = 0.0           # for debounce
        self._armed_at             = 0.0           # for trigger timeout
        self._door_opened_at       = 0.0           # for door open timer
        self._timer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event           = threading.Event()

        # Callbacks injected by main.py so this service stays decoupled
        self._on_motion_cb:  Optional[Callable] = None   # () → None
        self._on_reset_cb:   Optional[Callable] = None   # () → None
        self._enqueue:       Optional[Callable] = None   # (dict) → None

        self.connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_callbacks(self,
                      on_motion:  Callable,
                      on_reset:   Callable,
                      enqueue_fn: Callable):
        """
        Inject callbacks from main.py:
          on_motion()  → tell VideoWorker to arm recognition
          on_reset()   → tell VideoWorker recognition cycle is over
          enqueue_fn() → push WS event dict into the shared event queue
        """
        self._on_motion_cb = on_motion
        self._on_reset_cb  = on_reset
        self._enqueue      = enqueue_fn

    def start(self):
        """Open serial port and start reader + timer threads."""
        self._stop_event.clear()
        self._try_connect()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="arduino-reader"
        )
        self._timer_thread = threading.Thread(
            target=self._timer_loop, daemon=True, name="arduino-timer"
        )
        self._reader_thread.start()
        self._timer_thread.start()
        logger.info(f"ArduinoService started (port={SERIAL_PORT}, baud={BAUD_RATE})")

    def stop(self):
        """Signal threads to stop and close serial port."""
        self._stop_event.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        logger.info("ArduinoService stopped")

    def open_door(self):
        """
        Called by VideoWorker when access is granted.
        Sends OPEN to Arduino and transitions to DOOR_OPEN state.

        Race-condition safety: if the timer thread reset state to IDLE just
        before this call arrived (CPU inference took longer than TRIGGER_TIMEOUT),
        we still open the door as long as the last trigger was recent enough.
        """
        with self._lock:
            # Normal case — system is still ARMED or RECOGNIZING
            if self._state in (State.ARMED, State.RECOGNIZING):
                self._state = State.DOOR_OPEN
                self._door_opened_at = time.time()

            # Race-condition recovery — timeout fired just before recognition
            # completed. Allow opening if last trigger was within grace window.
            elif self._state == State.IDLE:
                grace = TRIGGER_TIMEOUT + 5.0   # 5-second grace on top of timeout
                since_trigger = time.time() - self._last_trigger_time
                if since_trigger <= grace:
                    logger.info(
                        f"open_door() race recovery: trigger was {since_trigger:.1f}s ago "
                        f"(within {grace}s grace) — opening anyway"
                    )
                    self._state = State.DOOR_OPEN
                    self._door_opened_at = time.time()
                else:
                    logger.warning(
                        f"open_door() ignored: state=IDLE, last trigger was {since_trigger:.1f}s ago "
                        f"(exceeds {grace}s grace window)"
                    )
                    return

            # Door already open or in cooldown — truly ignore
            else:
                logger.warning(f"open_door() ignored in state={self._state}")
                return

        self._send("OPEN")
        logger.info("OPEN sent to Arduino — door unlocking")
        self._push_event({
            "type":    "door_opened",
            "message": "Door unlocked — servo rotating to 90 degrees",
            "timestamp": self._ts(),
        })

    def deny_and_reset(self):
        """
        Called by VideoWorker when face is unknown or no face found in time.
        Resets back to IDLE without opening the door.
        """
        with self._lock:
            if self._state not in (State.ARMED, State.RECOGNIZING):
                return
            self._state = State.IDLE

        logger.info("Access denied — system reset to IDLE")
        self._push_event({
            "type":      "access_denied_reset",
            "message":   "No valid face — system disarmed",
            "timestamp": self._ts(),
        })
        if self._on_reset_cb:
            self._on_reset_cb()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_armed(self) -> bool:
        """VideoWorker checks this to decide whether to run recognition."""
        with self._lock:
            return self._state in (State.ARMED, State.RECOGNIZING)

    def set_recognizing(self):
        """Called by VideoWorker once it starts processing a frame after arming."""
        with self._lock:
            if self._state == State.ARMED:
                self._state = State.RECOGNIZING

    # ── Serial helpers ────────────────────────────────────────────────────────

    def _try_connect(self):
        try:
            import serial
            self._ser = serial.Serial(
                SERIAL_PORT, BAUD_RATE,
                timeout=1,
                write_timeout=2,
            )
            logger.info(f"Serial port opened: {SERIAL_PORT} @ {BAUD_RATE}")
            # Wait for Arduino READY message (sent by setup())
            deadline = time.time() + 5.0
            while time.time() < deadline:
                raw = self._ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line == "READY":
                        logger.info("Arduino READY received")
                        break
            else:
                logger.warning("Arduino READY not received within 5s — continuing anyway")
            self.connected = True
            logger.info(f"Arduino connected on {SERIAL_PORT}")
        except Exception as e:
            self.connected = False
            logger.warning(f"Arduino not available: {e}  (running in SOFTWARE-ONLY mode)")

    def _send(self, command: str):
        """Write a newline-terminated command to Arduino and flush immediately."""
        if self._ser and self._ser.is_open:
            try:
                payload = f"{command}\n".encode("ascii")
                self._ser.write(payload)
                self._ser.flush()
                logger.info(f"Arduino TX: {command}")
            except Exception as e:
                logger.error(f"Serial write error: {e}")
        else:
            logger.info(f"[SIMULATED] Arduino TX: {command}")

    # ── Reader thread — listens for Arduino messages, auto-reconnects ─────────

    def _reader_loop(self):
        RETRY_INTERVAL = 5.0   # seconds between reconnect attempts
        last_retry = 0.0

        while not self._stop_event.is_set():
            # ── Auto-reconnect if port was denied at startup or lost ──────────
            if not self.connected or not self._ser or not self._ser.is_open:
                now = time.time()
                if now - last_retry >= RETRY_INTERVAL:
                    last_retry = now
                    logger.info(f"Attempting to connect to Arduino on {SERIAL_PORT} ...")
                    self._try_connect()
                time.sleep(1)
                continue

            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                # Strip \r\n — Arduino on Windows sends \r\n
                line = raw.decode("utf-8", errors="ignore").strip().strip('\r\n').strip()
                if line:
                    logger.info(f"Arduino RX: [{line}]")
                    self._handle_arduino_message(line)
            except Exception as e:
                logger.error(f"Serial read error: {e} — marking disconnected")
                self.connected = False
                time.sleep(1)

    def _handle_arduino_message(self, line: str):
        """Parse messages from Arduino and drive the state machine."""
        if line == "MOTION_DETECTED":
            self._handle_ultrasonic_trigger()
        elif line == "DOOR_CLOSED":
            # Arduino confirms servo returned to 0°
            logger.info("Arduino confirmed door closed")
        elif line.startswith("PING"):
            self._send("PONG")

    def _handle_ultrasonic_trigger(self):
        now = time.time()
        with self._lock:
            # ── Debounce ──────────────────────────────────────────────────────
            if now - self._last_trigger_time < DEBOUNCE_SEC:
                logger.info("Ultrasonic trigger debounced (too fast)")
                return

            # ── Ignore during DOOR_OPEN or COOLDOWN ───────────────────────────
            if self._state in (State.DOOR_OPEN, State.COOLDOWN):
                logger.info(f"Ultrasonic trigger ignored — state={self._state}")
                return

            # ── Arm the system ────────────────────────────────────────────────
            self._last_trigger_time = now
            self._armed_at          = now
            self._state             = State.ARMED

        logger.info("Ultrasonic trigger received — system ARMED")
        self._push_event({
            "type":      "ultrasonic_triggered",
            "message":   "Person detected by ultrasonic sensor — recognition armed",
            "timestamp": self._ts(),
        })

        # Tell VideoWorker to start recognition
        if self._on_motion_cb:
            self._on_motion_cb()

    # ── Timer thread — handles timeouts, door close, cooldown reset ───────────

    def _timer_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.25)   # check 4× per second — responsive but not busy
            self._check_timers()

    def _check_timers(self):
        now = time.time()
        with self._lock:
            state = self._state

        if state == State.ARMED or state == State.RECOGNIZING:
            # ── Trigger timeout: nobody recognised within TRIGGER_TIMEOUT ──────
            with self._lock:
                armed_at = self._armed_at
            if now - armed_at > TRIGGER_TIMEOUT:
                logger.warning("Trigger timeout — no face recognised, resetting to IDLE")
                with self._lock:
                    self._state = State.IDLE
                self._push_event({
                    "type":      "trigger_timeout",
                    "message":   f"No face detected within {TRIGGER_TIMEOUT}s — system disarmed",
                    "timestamp": self._ts(),
                })
                if self._on_reset_cb:
                    self._on_reset_cb()

        elif state == State.DOOR_OPEN:
            # ── Door open timeout: close and enter cooldown ────────────────────
            with self._lock:
                door_opened_at = self._door_opened_at
            if now - door_opened_at > DOOR_OPEN_SEC:
                logger.info(f"Door open time elapsed ({DOOR_OPEN_SEC}s) — entering COOLDOWN")
                with self._lock:
                    self._state = State.COOLDOWN
                    self._door_opened_at = now   # reuse as cooldown start
                self._send("CLOSE")
                self._push_event({
                    "type":      "door_closed",
                    "message":   f"Door closed after {DOOR_OPEN_SEC}s — cooldown active",
                    "timestamp": self._ts(),
                })

        elif state == State.COOLDOWN:
            # ── Cooldown expired: back to IDLE ─────────────────────────────────
            with self._lock:
                cooldown_start = self._door_opened_at
            if now - cooldown_start > COOLDOWN_SEC:
                logger.info(f"Cooldown complete ({COOLDOWN_SEC}s) — system back to IDLE")
                with self._lock:
                    self._state = State.IDLE
                self._push_event({
                    "type":      "system_ready",
                    "message":   "System ready — waiting for next person",
                    "timestamp": self._ts(),
                })
                if self._on_reset_cb:
                    self._on_reset_cb()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _push_event(self, event: dict):
        """Push into the shared event queue (same one VideoWorker uses)."""
        if self._enqueue:
            self._enqueue(event)

    @staticmethod
    def _ts() -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

    def get_status(self) -> dict:
        with self._lock:
            state = self._state
            now   = time.time()
            extra = {}
            if state == State.DOOR_OPEN:
                extra["door_open_remaining"] = max(0, DOOR_OPEN_SEC - (now - self._door_opened_at))
            elif state == State.COOLDOWN:
                extra["cooldown_remaining"]  = max(0, COOLDOWN_SEC  - (now - self._door_opened_at))
            elif state in (State.ARMED, State.RECOGNIZING):
                extra["trigger_timeout_remaining"] = max(0, TRIGGER_TIMEOUT - (now - self._armed_at))
        return {
            "state":     state,
            "connected": self.connected,
            **extra,
        }


# ── Singleton — imported everywhere ──────────────────────────────────────────
arduino_service = ArduinoService()