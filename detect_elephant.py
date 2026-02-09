import cv2
import time
import requests
from collections import deque
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
RTSP_URL = "rtsp://admin:Admin%402026@192.168.68.110:554"

ESP32_ALERT_URL = "http://192.168.68.112/alert"
ESP32_CLEAR_URL = "http://192.168.68.112/clear"

CONFIDENCE = 0.30
FRAME_SKIP = 2
RESIZE_W, RESIZE_H = 800, 450

# MULTI-FRAME CONFIRMATION
CONFIRM_WINDOW = 7        # last 7 frames
CONFIRM_COUNT = 3         # 3 detections confirm elephant

# PRESENCE LOGIC
ABSENCE_RESET_SEC = 30    # 30 sec no detection → elephant gone
MIN_SIREN_TIME = 60       # siren minimum ON time (seconds)

# =====================
# LOAD MODEL
# =====================
model = YOLO("yolov8n.pt")

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
    results = model(frame, conf=CONFIDENCE, classes=[20])

    now = time.time()

    if len(results[0].boxes) > 0:
        detection_history.append(1)
        last_seen_time = now
        print("🐘 elephant detected (frame)")
    else:
        detection_history.append(0)

    # ===== ELEPHANT CONFIRM =====
    if (not elephant_present) and sum(detection_history) >= CONFIRM_COUNT:
        elephant_present = True
        siren_on_time = now
        print("🚨 ELEPHANT PRESENT → SIREN ON")

        if not alert_sent:
            try:
                requests.get(ESP32_ALERT_URL, timeout=2)
                print("📡 ESP32 ALERT SENT (SMS + SIREN)")
                alert_sent = True
            except:
                print("⚠ ESP32 not reachable")

    # ===== ELEPHANT LEFT =====
    if elephant_present:
        if (now - last_seen_time) > ABSENCE_RESET_SEC:
            if (now - siren_on_time) >= MIN_SIREN_TIME:
                try:
                    requests.get(ESP32_CLEAR_URL, timeout=2)
                    print("🔕 ESP32 CLEAR SENT (Siren OFF)")
                except:
                    print("⚠ ESP32 not reachable for CLEAR")

                elephant_present = False
                alert_sent = False
                detection_history.clear()
                print("🔕 ELEPHANT LEFT → SYSTEM RESET")

    time.sleep(0.005)
