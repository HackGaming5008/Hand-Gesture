import mediapipe as mp
import cv2
import numpy as np 
import pickle
import socket
import time

#---------------CONFIG------------------
UDP_IP = "127.0.0.1"
SEND_PORT = 4242
RECIVE_PORT = 4243
MODEL_PATH = 'spell_model.pkl'
HAND_MODEL = "model/hand_landmarker.task"

#---Load assets----
with open(MODEL_PATH, 'rb') as f:
    model, encoder = pickle.load(f)

# NETWORK
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((UDP_IP, RECIVE_PORT))
sock_recv.setblocking(False)

# Model Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def normalize_landmarks(landmark_list):
    base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z
    dist = ((landmark_list[9].x - base_x)**2 + (landmark_list[9].y - base_y)**2 + (landmark_list[9].z - base_z)**2)**0.5
    scale = dist if dist != 0 else 1
    flat_data = []
    for lm in landmark_list:
        flat_data.extend([(lm.x - base_x) / scale, (lm.y - base_y) / scale, (lm.z - base_z) / scale])
    return np.array(flat_data).reshape(1, -1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.IMAGE
)

cap = None
is_active = False
is_in_battle = False

with HandLandmarker.create_from_options(options) as landmarker:
    print("Script Active! Waiting for Godot commands...")

    while True:
        try:
            data, _ = sock_recv.recvfrom(1024)
            command = data.decode()
            
            if command == "START":
                is_active = True
                print("\nRecognition: ON")
            elif command == "STOP":
                is_active = False
                print("\nRecognition: OFF")
            elif command == "ENTRED_BATTLE": # Matching your Godot spelling
                if cap is None:
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 320)
                    cap.set(4, 240)
                    print("\nCamera: STARTING")
                    sock_send.sendto("Cam on".encode(), (UDP_IP, SEND_PORT))
                is_in_battle = True
            elif command == "EXIT_BATTLE":
                is_in_battle = False
                is_active = False # Safety: stop tracking if battle ends
                if cap:
                    cap.release()
                    cap = None
                print("\nCamera: SHUTDOWN")
                sock_send.sendto("Cam off".encode(), (UDP_IP, SEND_PORT))

        except BlockingIOError:
            pass

        # ONLY process if both battle is on AND recognition is active AND cap exists
        if is_active and is_in_battle and cap is not None:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    features = normalize_landmarks(landmarks)
                    
                    probs = model.predict_proba(features)[0]
                    confidence = np.max(probs) * 100

                    if confidence >= 95:            
                        prediction = np.argmax(probs)
                        current_sign = encoder.inverse_transform([prediction])[0]
                        sock_send.sendto(current_sign.encode(), (UDP_IP, SEND_PORT))
                        print(f"Sent: {current_sign} ({confidence:.1f}%)", end="\r")

        time.sleep(0.01)