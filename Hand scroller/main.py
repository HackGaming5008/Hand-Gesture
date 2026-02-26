import mediapipe as mp
import cv2
import time
import math
import pyautogui

from mediapipe.tasks import python 
from mediapipe.tasks.python import vision

######################## OPTIONS ############################
camShow_mode = False
tick_Mode = True

# step scroll
step_threshold = 0.04
stepatOnce = 100

# smooth scroll
scroll_delay = 0.06 

######################## Model Step Up ###########################

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create the hand landmarker instances with the live stream mode:

class HandTracker:
	def __init__(self):
		self.result = None

	def callback(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
		self.result = result	

tracker = HandTracker()
model_path = 'model/hand_landmarker.task'


pyautogui.FAILSAFE = True



def dist_fromto(target, fixedtar):
	dist = math.hypot((target.x - fixedtar.x), (target.y - fixedtar.y))
	return dist


options = HandLandmarkerOptions(

	base_options=BaseOptions(model_asset_path=model_path),
	running_mode=VisionRunningMode.LIVE_STREAM,
	result_callback=tracker.callback,

)

cap = cv2.VideoCapture(0)

scroll_mode = False
counter = 0

hand_points = [0, 6, 8, 10, 12, 14, 16, 18, 20]

fingers = {  # tip, mid
	"index": (8,6),
	"middle": (12, 10),
	"ring": (16, 14),
	"pinky": (20, 18)
}



prev_y = None
scroll_speed = 0

# smooth scroll
last_scroll_time = 0

# step scroll
movement_accumulator = 0



######################################### MAIN LOOP ###################################


with HandLandmarker.create_from_options(options) as landmarker:
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break

		h, w, _= frame.shape
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb

		mp_image = mp.Image(

			image_format = mp.ImageFormat.SRGB,
			data=rgb_frame

		)

		timestamp = int(time.time() * 1000)
		landmarker.detect_async(mp_image, timestamp)

		if tracker.result and tracker.result.hand_landmarks:
			for landmarks in tracker.result.hand_landmarks:

				current_y = landmarks[12].y # middle tip

				finger_states = {}

				for name, (tip_id, mid_id) in fingers.items():
					tip = landmarks[tip_id]
					mid = landmarks[mid_id]

					wrist = landmarks[0]

					tip_dist = dist_fromto(tip, wrist)
					mid_dist = dist_fromto(mid, wrist)

					finger_states[name] = tip_dist > mid_dist + 0.01

				scroll_gesture = (
					finger_states["index"] and 
					finger_states["middle"] and 
					not finger_states["ring"] and 
					not finger_states["pinky"]
				)

				if scroll_gesture:
					counter = min(counter + 1, 15)
				else:
					counter = max(counter - 1, 0)

				
				
                ########## Draw point on joint and tip ##############
				if camShow_mode:
					for i in hand_points:
						point = landmarks[i] # joint points and waist
						# curr_y = point.y

						cx, cy = int(point.x * w), int(point.y * h)
						cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
				else:
					pass

		else:
			counter = max(counter - 1, 0)


###################### Scroll mode ON OFF #####################
		if counter >= 10:
			scroll_mode =True
		elif counter <= 5:
			scroll_mode = False
			movement_accumulator = 0
			prev_y = None

###################### Scroll Logic #############################


		if scroll_mode:
			
			current_time = time.time()

			if prev_y is not None:
				delta = current_y- prev_y

				if tick_Mode:

					movement_accumulator += delta

					if movement_accumulator > step_threshold:
						pyautogui.scroll(stepatOnce)
						movement_accumulator = 0
					elif movement_accumulator < -step_threshold:
						pyautogui.scroll(-stepatOnce)
						movement_accumulator = 0

				else:

					scroll_speed = int(max(min(delta * 600, 500), -500))

					if abs(delta) > 0.003 and current_time - last_scroll_time > scroll_delay:
						pyautogui.scroll(scroll_speed)
						last_scroll_time = current_time

			prev_y = current_y







###################### Cam Logic ####################

		if camShow_mode:
			cv2.imshow("Camera", frame)
		else:
			pass

		if cv2.waitKey(1) & 0xFF == 27: #exit if escap is pressed
			break

cap.release()
cv2.destroyAllWindows()
