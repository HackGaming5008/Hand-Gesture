import mediapipe as mp
import cv2
import time
import math

from mediapipe.tasks import python 
from mediapipe.tasks.python import vision


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




def dist_fromto(target, fixedtar):
	dist = math.hypot((target.x - fixedtar.x), (target.y - fixedtar.y))
	return dist


options = HandLandmarkerOptions(

	base_options=BaseOptions(model_asset_path=model_path),
	running_mode=VisionRunningMode.LIVE_STREAM,
	result_callback=tracker.callback,
)

cap = cv2.VideoCapture(0)





######################################### MAIN LOOP ###################################


with HandLandmarker.create_from_options(options) as landmarker:
	while cap.isOpened():

		pass