# import the necessary packages
import numpy as np
# import numpy.random.common
# import numpy.random.bounded_integers
# import numpy.random.entropy
import argparse
import imutils
# import time
import cv2
import os
from imutils.video import FPS
from collections import Counter
from datetime import *
import pytz 
import pickle


args = {}
args["confidence"] = 0.5
args["threshold"]  = 0.3
args["skip_frames"] = 60

totalFrames = 0

args["output"] = "output/test_01.avi"

# derive the paths to the YOLO weights and model configuration
configPath = "model.cfg"
weightsPath = "model.weights"

# configPath = "yolov3-tiny.cfg"
# weightsPath = "yolov3-tiny.weights"

calculations = {}
calculations_hour = {}

# load the COCO class labels our YOLO model was trained on
LABELS = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter', 'bench','bird',
 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
 'kite','baseball bat','baseball glove','skateboard','surfboard', 'tennis racket','bottle','wine glass','cup','fork', 'knife','spoon',
 'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet',
 'tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
# print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
# vs = cv2.VideoCapture("test_day.avi")

vs = cv2.VideoCapture()
vs.open("rtsp://admin:admin123@123.231.62.102:554/Streaming/channels/101")


writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	# print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	# print("[INFO] could not determine # of frames in video")
	# print("[INFO] no approx. completion time can be provided")
	total = -1

fps = FPS().start()
data = []

temp_time = 1

tz_INDIA = pytz.timezone('Asia/Kolkata')
date_data = datetime.now(tz_INDIA) 
day = date_data.day
hour = date_data.hour
minute = date_data.minute


# loop over frames from the video file stream
while True:

	# tz_INDIA = pytz.timezone('Asia/Kolkata')  
	datetime_INDIA = datetime.now(tz_INDIA) 
	if (datetime_INDIA.year == 2021) and (datetime_INDIA.month == 12):
		print(f"Your license period has expired")
		break 

	# print(f"BEGINNING: The temp time is {temp_time}")
	if datetime_INDIA.hour!=temp_time:
		data=[]
	
	# print(f"The data is {data}")

	try:
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream. Continue the program for the next iteration.
		if not grabbed:
			print(f"There is no grabbing of frames")
			continue

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities

		if totalFrames % args["skip_frames"] == 0:

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			# start = time.time()
			layerOutputs = net.forward(ln)
			# end = time.time()

			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []


			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability)
					# of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > args["confidence"]:
						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])
			# print(f"The idxs are {idxs} and type is {type(idxs)} ")
			# print(f"The flattened idxs are {idxs.flatten()} and type is {type(idxs.flatten())} ")

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# print(f"The i is {i}")
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					# print(f"The labels are {LABELS[classIDs[i]]}")
					data.append(LABELS[classIDs[i]])
					data_c = Counter(data)
					data_c = {k: v for k, v in data_c.items() if k in "cars" or k in "truck" or k in "bus" or k in "person" or k in "motorbike"}
					print(f"The data is {data_c}")
					# tz_INDIA = pytz.timezone('Asia/Kolkata')  
					# datetime_INDIA = datetime.now(tz_INDIA) 
					# calculations[datetime_INDIA.day] = {datetime_INDIA.hour: data_c}
					calculations_hour[datetime_INDIA.hour] = data_c
					calculations[datetime_INDIA.day] = calculations_hour
					# calculations_minute[datetime_INDIA.minute] = data_c
					# calculations[datetime_INDIA.day] = calculations_minute
					print(f"OTS count: {calculations}")
					pickle.dump(calculations, open(f'ots_count_{day}_{hour}_{minute}.p', 'wb'))
					temp_time = datetime_INDIA.hour
					# cv2.putText(frame, text, (x, y - 5),
					# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		# temp_time = datetime_INDIA.hour
		# print(f"END: The temp time is {temp_time}")
		totalFrames += 1
		fps.update()
	except Exception as ex:
		template = "An exception of type {0} occurred. Arguments:\n{1!r}"
		message = template.format(type(ex).__name__, ex.args)
		print (message)

# stop the timer and display FPS information
try:
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# release the file pointers
	print("[INFO] cleaning up...")
except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
finally:
	print("There seems to be some problem with the feed")
	vs.release()