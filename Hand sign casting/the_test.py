import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- SETTINGS ---
MODEL_PATH = 'spell_model.pkl'
HAND_MODEL = 'model/hand_landmarker.task'

# --- LOAD BRAIN ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model, encoder = pickle.load(f)
    print(f"Loaded brain with classes: {list(encoder.classes_)}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def normalize_landmarks(landmark_list):
    # THE PRO MATH: Wrist-relative + Middle MCP Scaling
    base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z
    
    # Distance between Wrist (0) and Middle MCP (9)
    dist = ((landmark_list[9].x - base_x)**2 + 
            (landmark_list[9].y - base_y)**2 + 
            (landmark_list[9].z - base_z)**2)**0.5
    scale = dist if dist != 0 else 1

    flat_data = []
    for lm in landmark_list:
        flat_data.extend([
            (lm.x - base_x) / scale, 
            (lm.y - base_y) / scale, 
            (lm.z - base_z) / scale
        ])
    return np.array(flat_data).reshape(1, -1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)
# Capping at 15 FPS so your i3 doesn't have a stroke
cap.set(cv2.CAP_PROP_FPS, 15)

print("\n--- STANDBY FOR SPELLCASTING ---")
print("Show your hand to the camera. Press 'Q' to quit.\n")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        current_sign = "None"
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            features = normalize_landmarks(landmarks)
            
            # Prediction
            prediction = model.predict(features)[0]
            current_sign = encoder.inverse_transform([prediction])[0]
            
            # Optional: Get probability (how sure the brain is)
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100
            
            # Print to console (using \r to keep it on one line)
            print(f"Detecting: {current_sign.upper()} | Confidence: {confidence:.1f}%          ", end="\r")

            # Draw landmarks on screen for visual feedback
            for lm in landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        else:
            print("No hand detected...                                     ", end="\r")

        cv2.imshow("Sorcerer Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()