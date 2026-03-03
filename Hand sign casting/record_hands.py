import cv2
import mediapipe as mp
import csv
import numpy as np
import time


# Setup the "Modern" Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


LABEL = "neutral"
DATA_FILE = "hand_data.csv"
model_path = 'model/hand_landmarker.task'

# modify this from gpt and delete the data file and redo again

def normalize_landmarks(landmark_list):
    # 1. Wrist-relative (Crucial!)
    base_x, base_y, base_z = landmark_list[0].x, landmark_list[0].y, landmark_list[0].z
    
    # 2. Calculate distance between Wrist (0) and Middle MCP (9)
    # Using the math ChatGPT suggested for better consistency
    dist = ((landmark_list[9].x - base_x)**2 + 
            (landmark_list[9].y - base_y)**2 + 
            (landmark_list[9].z - base_z)**2)**0.5
    scale = dist if dist != 0 else 1

    flat_data = []
    for lm in landmark_list:
        # Subtract wrist and DIVIDE by the scale
        flat_data.extend([
            (lm.x - base_x) / scale, 
            (lm.y - base_y) / scale, 
            (lm.z - base_z) / scale
        ])
    
    return flat_data


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)



cap = cv2.VideoCapture(0)

print(f"Press 'r' to record data for: {LABEL}")
print("Press 'Q' to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
	print(f"Recording for: {LABEL} | Press 'R' to save, 'Q' to quit.")

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break

		frame = cv2.flip(frame, 1)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

		result = landmarker.detect(mp_image)

		if result.hand_landmarks:
			for landmarks in result.hand_landmarks:

				for lm in landmarks:
					x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
					cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


				if cv2.waitKey(1) & 0xFF == ord('r'):
					data = normalize_landmarks(landmarks)
					with open(DATA_FILE, 'a', newline='') as f:
						writer = csv.writer(f)

						writer.writerow([LABEL, *data])
					print(f"Saved {LABEL} sample!")


		cv2.imshow("Data collection -  Press 'R' to record", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):break

cap.release()
cv2.destroyAllWindows()

