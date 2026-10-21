# Face Recognition Backend

A production-ready face recognition backend built with FastAPI and ArcFace for real-time identity verification and access control. The system performs live face recognition, motion detection, liveness detection, WebSocket streaming, role-based access control, and hardware integration for smart access management.

---

## Features

- Real-time Face Recognition
- ArcFace (InsightFace) Embeddings
- FastAPI REST APIs
- WebSocket Live Streaming
- JWT Authentication
- Motion Detection
- Liveness Detection
- MongoDB Atlas Integration
- User Registration
- Unknown Person Alerts
- Access Logs
- Role-Based Access Control
- Arduino Door Control Integration
- Embedding Cache
- Background Recognition Worker
- Live System Statistics

---

## Tech Stack

- Python
- FastAPI
- InsightFace (ArcFace)
- OpenCV
- NumPy
- PyTorch
- MongoDB Atlas
- WebSocket
- JWT
- Uvicorn
- Pydantic
- Python Dotenv

---

## Project Structure

```
project/
│
├── services/
│   ├── arduino_service.py
│
├── logs/
├── unknown_alerts/
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## Core Modules

### Authentication

- JWT Login
- Token Verification
- Protected APIs

### Face Recognition

- ArcFace Embeddings
- Face Matching
- Similarity Scoring
- Embedding Cache

### Recognition Pipeline

- Webcam Capture
- Motion Detection
- Face Detection
- Liveness Detection
- Face Recognition
- Access Decision
- Event Broadcasting

### Access Control

- Admin Access
- Guest Access Schedule
- Unknown Person Detection
- Cooldown Protection

### Database

MongoDB stores:

- Registered Users
- Face Embeddings
- Access Logs
- Unknown Person Alerts

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

Navigate to the project

```bash
cd your-repository
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
MONGODB_URI=

DB_NAME=

JWT_SECRET=

ADMIN_USERNAME=

ADMIN_PASSWORD=

SIMILARITY_THRESHOLD=0.45

WEBCAM_INDEX=0

ARDUINO_ENABLED=true

ARDUINO_PORT=COM3

ARDUINO_BAUD=9600

GUEST_START_HOUR=9

GUEST_END_HOUR=18
```

---

## Running the Server

```bash
python main.py
```

or

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API will be available at

```
http://localhost:8000
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/auth/login` | Admin login |
| GET | `/auth/me` | Current user |
| POST | `/register` | Register a new face |
| POST | `/recognize` | Single image recognition |
| GET | `/users` | List registered users |
| DELETE | `/user` | Delete a user |
| GET | `/logs` | Access logs |
| GET | `/alerts` | Unknown person alerts |
| GET | `/stats` | System statistics |
| GET | `/worker/status` | Recognition worker status |
| POST | `/worker/control` | Enable/disable recognition |
| GET | `/health` | Health check |
| POST | `/reload-cache` | Reload embeddings |

---

## WebSocket

Live endpoint

```
ws://localhost:8000/ws/live
```

Provides:

- Live Camera Frames
- Recognition Events
- Motion Detection
- System Status
- Access Events
- Unknown Person Alerts

---

## Recognition Workflow

1. Capture webcam frames
2. Detect motion
3. Detect face
4. Verify liveness
5. Generate ArcFace embedding
6. Compare with registered embeddings
7. Check similarity threshold
8. Apply access rules
9. Log event
10. Broadcast via WebSocket

---

## Security Features

- JWT Authentication
- Protected Endpoints
- Similarity Threshold
- Liveness Detection
- Motion Detection
- Access Cooldown
- Unknown Person Alerts
- Role-Based Authorization

---

## Database Collections

### users

Stores:

- User ID
- Name
- Role
- Face Embeddings

### logs

Stores:

- Access Status
- User Details
- Similarity Score
- Timestamp

### alerts

Stores:

- Unknown Person Image
- Similarity Score
- Alert Timestamp

---

## Hardware Integration

Supports Arduino-based access control.

Features include:

- Ultrasonic Motion Trigger
- Door Unlock Control
- Recognition Trigger
- Automatic Reset
- Live Hardware Status

---

## Future Improvements

- Multi-Camera Support
- GPU Acceleration
- Face Mask Detection
- Face Anti-Spoofing Models
- Email Notifications
- Mobile Application
- Docker Deployment
- Kubernetes Deployment
- Cloud Storage Integration
- Audit Dashboard

---

## License

This project is intended for educational, research, and development purposes.
