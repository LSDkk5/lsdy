from lsdy import LsdY
from time import time
from glob import glob
import numpy as np
import cv2 as cv
import os

class DetectObjectsFromImage(LsdY):
    def start(self, showp):
        print("[INFO] Loading files from disk...")
        self.__net = cv.dnn.readNetFromDarknet(self.configsPath, self.weightsPath)
        self.__font = cv.FONT_HERSHEY_SIMPLEX
        self.__font_scale = 0.5

        image = ''
        imgExtList = {'jpg', 'jpeg', 'png', 'PNG', 'JPG', 'JPEG'}
        for imgExt in imgExtList:
            imageList = glob(f'{self.inputPath}/*.{imgExt}')
            for img in imageList:
                imgName = os.path.basename(img)
                image = cv.imread(img)
                (H, W) = image.shape[:2]

                ln = self.__net.getLayerNames()
                ln = [ln[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]
                blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.__net.setInput(blob)
                start = time()
                layerOutputs = self.__net.forward(ln)

                boxes = []
                confidences = []
                classIDs = []

                for out in layerOutputs:
                    for detection in out:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if confidence>self.CONFIDENCE:
                            box = detection[0:4]*np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype('int')

                            x = int(centerX-(width/2))
                            y = int(centerY-(height/2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                if len(idxs)>0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, _) = (boxes[i][2], boxes[i][3])
                        color = [int(c) for c in self.set_colors(42)[classIDs[i]]]
                        end = time()
                        if showp:
                            percentage = str(confidences[i]-int(confidences[i])).split('.')[1]
                            text = f'{self.load_labels()[classIDs[i]]} {percentage[:2]}%'
                            print(f'[DETECTED] {self.load_labels()[classIDs[i]].upper()}: time {round(abs(start-end), 2)}s        | probability: {percentage[:2]}%')
                        else: 
                            text = f'{self.load_labels()[classIDs[i]]}'
                            print(f'[DETECTED] {self.load_labels()[classIDs[i]].upper()}: time {round(abs(start-end), 2)}s')
                        cv.putText(image, text, (x, y - 5), self.__font, fontScale=self.__font_scale,
                                    color=color, thickness=2, lineType=cv.LINE_4)
                        cv.rectangle(image, (x, y), (x+w,  y+w), color, 2)
                cv.imwrite(f'{self.outputPath}/{imgName}', image)
        scriptEnd = time()
        print(f'[INFO] Total time {round(abs(self.scriptStart-scriptEnd), 2)}s')