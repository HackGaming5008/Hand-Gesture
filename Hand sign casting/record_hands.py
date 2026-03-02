import cv2
import mediapipe as mp
import csv
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands = 1, min_detection_confidence=0.7)


LABLE = "fire"
DATA_FILE = "hand_data.csv"


def normalize_landmarks(landmarks):

	temp_list = []

	for lm in landmarks.landmark:
		temp_list.append([lm.x, lm.y, lm.z])

	base_x, base_y, base_z = temp_list[0] #coordinates of the wrist
	for i in range(len(temp_list)):
		temp_list[i][0] -= base_x
		temp_list[i][1] -= base_y
		temp_list[i][2] -= base_z

	flat_list = np.array(temp_list).flatten()

	max_value = np.abs(flat_list).max()

	if max_value != 0:
		flat_list = flat_list / max_value

	return flat_list

cap = cv2.VideoCapture(0)

print(f"Press 'r' to record data for: {LABLE}")
print("Press 'Q' to quit.")

while cap.isOpened():
	ret, frame = cap.read()
	if not ret: break

	frame = cv2.flip(frame, 1)
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = hands.process(rgb_frame)

	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:

			mp.solutions.draw_utils.draw_landmarks(frame, hand_landmarks, mp.hands.HAND_CONNECTIONS)

			if cv2.waitKey(1) & 0xFF == ord('r'):
				data = normalize_landmarks(hand_landmarks)
				with open(DATA_FILE, 'a', newline='') as f:
					writer = csv.writer(f)

					writer.writerow([LABLE, *data])
				print(f"Saved {LABLE} sample!")


	cv2.imshow("Data collection -  Press 'R' to record", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):break

cap.release()
cv2.destroyAllWindows()

