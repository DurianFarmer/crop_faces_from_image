# how to execute
# ex) if you want to make face numbering start from 10 (default: 1), execute as follows:
# python crop_faces_from_image.py -n 10

# import numpy
import numpy as np

# module for system control
import os
import argparse

# module for webcam and capturing a webcam screenshot
import cv2
from PIL import Image

# modules for face recognition
import face_recognition
import pickle

# construct the argument parser and parse the arguments
INPUT = 'input' # or 'input_wom' or 'input_wm'
OUTPUT = 'output' # or 'output_wom' or 'output_wm'
#INPUT = 'input' # directory containing images that have faces
#OUTPUT = 'output' # directory of cropped faces
MODEL = 'face_detector'
CONFIDENCE = 0.5

# load the serialized face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([MODEL, "deploy.prototxt"])
weightsPath = os.path.sep.join([MODEL, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# set the starting number of the output faces
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', type = int, default = 1, help = 'starting number of the output faces')
N = vars(ap.parse_args())['number']
n = vars(ap.parse_args())['number']

for file in os.listdir(INPUT):
    # load the input image and grab the image spatial dimensions
    image = cv2.imread(f'{INPUT}/{file}')
    
    # wc - error pass
    if image is None:
        continue

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # setting varaibles
    (h, w) = image.shape[:2]
    (prev_startX, prev_startY, prev_endX, prev_endY) = (0, 0, 0, 0)

    # loop over the detections for each image
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > CONFIDENCE:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") # pixels of image

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                            
            face = image[startY: endY, startX: endX].copy()

            # capture wrong faces
            if (startX < prev_endX and endX > prev_endX and startY < prev_endY and endY > prev_endY):
                # case 1: previous is the face, current is the left-side neck
                prev_startX, prev_startY, prev_endX, prev_endY = startX, startY, endX, endY
                continue
            elif  (endX > prev_startX and endX < prev_endX and endY > prev_startY and endY < prev_endY):
                # case 2: previous is the left-side neck, current is the face
                prev_startX, prev_startY, prev_endX, prev_endY = startX, startY, endX, endY                
                # rewrite image
                cv2.imwrite(f'{OUTPUT}/face{n-1}.jpg', face)
                # add 1 to n
                n += 1
            else:
                try:                
                    # print('n: ', n)
                    # print('(startX, endX, startY, endY): ', (startX, endX, startY, endY))
                    cv2.imwrite(f'{OUTPUT}/face{n}.jpg', face)
                    # add 1 to n
                    n += 1
                except Exception:
                    pass
                finally:
                    prev_startX, prev_startY, prev_endX, prev_endY = startX, startY, endX, endY

print(f'complete!! {n-N} faces returned')