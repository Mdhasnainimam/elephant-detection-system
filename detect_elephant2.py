import cv2
import time
import requests
from collections import deque
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
RTSP_URL = "rtsp://admin:Admin%402026@192.168.1.122:554"

# Render Cloud URLs
RENDER_ALERT_URL = "https://elephant-detection-2380.onrender.com/alert"
RENDER_CLEAR_URL = "https://elephant-detection-2380.onrender.com/clear"

# ESP32 (local siren – optional)
ESP32_ALERT_URL = "http://192.168.1.103/alert"
ESP32_CLEAR_URL = "http://192.168.1.103/clear"

# =====================
# DETECTION TUNING (IMPORTANT)
# =====================
CONFIDENCE = 0.20            # lower confidence for partial elephant
FRAME_SKIP = 1               # detect every frame (no skip)

RESIZE_W, RESIZE_H = 1280, 720  # HIGH resolution for occlusion cases

# MULTI-FRAME CONFIRMATION
CONFIRM_WINDOW = 10          # last 10 frames
CONFIRM_COUNT = 4            # 4 detections confirm elephant

# PRESENCE LOGIC
ABSENCE_RESET_SEC = 30       # elephant gone timeout
MIN_SIREN_TIME = 60          # minimum siren ON time

LOCATION_NAME = "Delhi Forest"

# =====================
# LOAD STRONGER MODEL
# =====================
# yolov8m is MUCH better for partial / behind-tree detection
model = YOLO("yolov8m.pt")

# =====================
# VIDEO CAPTURE
# =====================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Camera stream open nahi hua")
    exit()

print("✅ Camera connected, detection started")

# =====================
# STATE VARIABLES
# =====================
frame_count = 0
detection_history = deque(maxlen=CONFIRM_WINDOW)

elephant_present = False
alert_sent = False
siren_on_time = 0
last_seen_time = 0

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    # Elephant class = 20 (COCO)
    results = model(frame, conf=CONFIDENCE, classes=[20])

    now = time.time()

    if len(results[0].boxes) > 0:
        detection_history.append(1)
        last_seen_time = now
        print("🐘 elephant detected (frame)")
    else:
        detection_history.append(0)

    # =====================
    # ELEPHANT CONFIRMED
    # =====================
    if (not elephant_present) and sum(detection_history) >= CONFIRM_COUNT:
        elephant_present = True
        siren_on_time = now
        print("🚨 ELEPHANT PRESENT → ALERT TRIGGERED")

        if not alert_sent:
            # ---- CLOUD ALERT (WhatsApp) ----
            try:
                r = requests.post(
                    RENDER_ALERT_URL,
                    json={
                        "event": "ELEPHANT_DETECTED",
                        "location": LOCATION_NAME,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    timeout=5
                )
                print("☁ Render ALERT sent:", r.status_code)
            except Exception as e:
                print("⚠ Render alert failed:", e)

            # ---- LOCAL SIREN (ESP32) ----
            try:
                requests.get(ESP32_ALERT_URL, timeout=2)
                print("📡 ESP32 ALERT SENT (SIREN ON)")
            except:
                print("⚠ ESP32 not reachable")

            alert_sent = True

    # =====================
    # ELEPHANT LEFT
    # =====================
    if elephant_present:
        if (now - last_seen_time) > ABSENCE_RESET_SEC:
            if (now - siren_on_time) >= MIN_SIREN_TIME:
                print("🔕 ELEPHANT LEFT → CLEAR ALERT")

                # ---- CLOUD CLEAR ----
                try:
                    r = requests.post(
                        RENDER_CLEAR_URL,
                        json={
                            "event": "ELEPHANT_CLEARED",
                            "location": LOCATION_NAME,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")
                        },
                        timeout=5
                    )
                    print("☁ Render CLEAR sent:", r.status_code)
                except Exception as e:
                    print("⚠ Render clear failed:", e)

                # ---- LOCAL SIREN OFF ----
                try:
                    requests.get(ESP32_CLEAR_URL, timeout=2)
                    print("📡 ESP32 CLEAR SENT (SIREN OFF)")
                except:
                    print("⚠ ESP32 not reachable for CLEAR")

                elephant_present = False
                alert_sent = False
                detection_history.clear()

    time.sleep(0.005)