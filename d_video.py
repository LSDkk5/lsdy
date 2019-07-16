from lsdy import LsdY
from glob import glob
import numpy as np
import time
import cv2
import os

class DetectObjectsFromVideo(LsdY):
	def start(self, live, showp):
		print("[INFO] loading files from disk...")
		net = cv2.dnn.readNetFromDarknet(self.configsPath, self.weightsPath)
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		if live:
			vs = cv2.VideoCapture(0)
		else:
			vidPath = glob(f'{self.inputPath}/*.mp4')[0]
			vs = cv2.VideoCapture(vidPath)
		fps = vs.get(cv2.CAP_PROP_FPS)
		writer = None
		(W, H) = (None, None)
		try:
			prop = cv2.CAP_PROP_FRAME_COUNT
			total = int(vs.get(prop))
			print(f"[INFO] {total} total frames in video")
			print(f"[INFO] frame rate: {round(fps, 3)}fps")
		except:
			print("[INFO] could not determine # of frames in video")
			print("[INFO] no approx. completion time can be provided")
			total = -1

		while True:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			boxes = []
			confidences = []
			classIDs = []

			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					if confidence > self.CONFIDENCE:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE, self.TRESHOLD)

			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in self.set_colors(42)[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					percentage = str(confidences[i]-int(confidences[i])).split('.')[1]
					if showp:
						text = f'{self.load_labels()[classIDs[i]]} {percentage[:2]}%'
					else:
						text = f'{self.load_labels()[classIDs[i]]}'
					cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(f"{self.outputPath}/{os.path.splitext(os.path.basename(vidPath))[0]}.avi", 
						fourcc, 30, (frame.shape[1], frame.shape[0]), True)
				print()
				if total > 0:
					elap = round((end - start), 2)
					print(f"[INFO] single frame took {elap} seconds")
					print(f"[INFO] estimated total time to finish: {round((elap*total), 2)}s")
			writer.write(frame)
		print("[INFO] cleaning up...")
		if live:
			cv2.imshow('live object detection', frame)
		else:	writer.release()
		vs.release()