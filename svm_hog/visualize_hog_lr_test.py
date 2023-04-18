from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from sklearn import *
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob


#Define HOG Parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)
threshold = .3

model = joblib.load('svm_hog/models/lr_model.dat')

# define the sliding window:
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


# Test the trained classifier on an image below!
scale = 0
detections = []
img= cv2.imread("INRIAPerson/Test/pos/crop_000007.png")
img= cv2.resize(img,(400,256)) 

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.25
for resized in pyramid_gaussian(img, downscale=1.25):
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
        print("window shape: ",window.shape, "resized : ", resized.shape)
        if window.shape[0] != winH or window.shape[1] !=winW: 
            continue
        window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2') 
        fds = fds.reshape(1, -1) 
        pred = model.predict(fds) 
        
        if pred == 1:
            if model.decision_function(fds) > 0.6: 
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), 
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = img.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) 
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
        
for(x1, y1, x2, y2) in pick:
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(clone,'Person',(x1-2,y1-2),1,0.75,(121,12,34),1)

print("Clone shape: ", clone.shape)
cv2.imshow('Person Detection',clone)
#press esc to exit
cv2.waitKey(0)
cv2.destroyAllWindows()