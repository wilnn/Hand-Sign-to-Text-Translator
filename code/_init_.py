##########################
#
# This file is the main hand sign translator program. It will translate the hand sign capture via camera to text 
# or action on screen.
#
##########################


import numpy as np
import cv2
import os
from dotenv import load_dotenv
import numpy as np
import keras
import mediapipe as mp
import tensorflow as tf
import time

load_dotenv()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
char = ''
nodetect = True

# configuring mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# create a hand landmarker instance with the image mode
options = HandLandmarkerOptions(
base_options=BaseOptions(model_asset_path=os.getenv('modelPath')),
running_mode=VisionRunningMode.IMAGE)

# create a hand lanmark detector instance
landmarker = HandLandmarker.create_from_options(options)

# load the model
model = tf.keras.models.load_model('models/hand_sign_translator.keras')
# freeze the dropout layer for more consistance result
for layer in model.layers:
    if type(layer) == keras.src.layers.regularization.dropout.Dropout:
        layer.training = False
time1 = time.perf_counter()

# start capturing
while True:
    if cv2.waitKey(1) == ord('q'):
        break 
    ret, frame = cap.read()

    # height and width of frame
    width = int(cap.get(3))
    height = int(cap.get(4))

    canvas = np.ones((height+50, width, 3), dtype=np.uint8)*255

    canvas[:height, :] = frame

    framee = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB format

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framee)

    landmarker_result = landmarker.detect(mp_image)
    
    # handle the case when can not detect hand in the image
    if not landmarker_result or not landmarker_result.hand_landmarks:
        if nodetect == False:
            img = cv2.putText(canvas, char, (10, height+40), font, 1.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', img)
        else:
            cv2.imshow('frame', canvas)
        continue
        
        
    # get the corners for the bounding box of the hand by finding
            # bigest and smallest x and y coordinate of the landmark
    for i in range(len(landmarker_result.hand_landmarks[0])):
        #img = cv2.circle(img, (round(landmarker_result.hand_landmarks[0][i].x*img.shape[1]), 
                                #round(landmarker_result.hand_landmarks[0][i].y*img.shape[0])),
                                #5, (0, 255, 0), thickness=-1) #draw circle on landmarks
        if i == 0:
            min_x = landmarker_result.hand_landmarks[0][i].x
            max_x = landmarker_result.hand_landmarks[0][i].x
            min_y = landmarker_result.hand_landmarks[0][i].y
            max_y = landmarker_result.hand_landmarks[0][i].y
        
        else:
            if landmarker_result.hand_landmarks[0][i].x < min_x:
                min_x = landmarker_result.hand_landmarks[0][i].x
            elif landmarker_result.hand_landmarks[0][i].x > max_x:
                max_x = landmarker_result.hand_landmarks[0][i].x
            if landmarker_result.hand_landmarks[0][i].y < min_y:
                min_y = landmarker_result.hand_landmarks[0][i].y
            elif landmarker_result.hand_landmarks[0][i].y > max_y:
                max_y = landmarker_result.hand_landmarks[0][i].y

    # convert to pixel value
    min_y = round((min_y -0.017) * frame.shape[0])
    max_y = round((max_y + 0.017) * frame.shape[0])
    min_x = round((min_x - 0.017) * frame.shape[1])
    max_x = round((max_x + 0.017) * frame.shape[1])
    
    hand = frame[min_y:max_y, min_x:max_x]

    # handle the problem when the image cropped is empty
    if hand is None or hand.shape[0] == 0 or hand.shape[1] == 0:
        if char is not None:
            img = cv2.putText(canvas, char, (10, height+40), font, 1.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', img)
        else:
            cv2.imshow('frame', canvas)
        continue

    hand = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
    hand = cv2.resize(hand, (190,190))

    # draw the rectangle around the hand
    frame = cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (0, 255, 0), 5)

    # the program will check for handsign in the video every 3 seconds
    time2 = time.perf_counter()
    if time2 - time1 < 1: # the lower the number, the faster the program will check for hand sign. Adjust this number to suit with your signing speed
        if nodetect == False:
            img = cv2.putText(canvas, char, (10, height+40), font, 1.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', img)
        else:
            cv2.imshow('frame', canvas)
        continue
    else:
        time1 = time2
    
    # predict the hand sign 
    img = (np.expand_dims(hand,0))
    predict = model.predict(img/255.0, verbose=0)
    # not output if confident score is less than 99%
    if np.max(predict[0]) < 0.99:
        nodetect = True
    elif np.argmax(predict[0]) == 26:
        char = char[:-1]
        nodetect = False
    elif np.argmax(predict[0]) == 27:
        char +=' '
        nodetect = False
    else:
        char += chr(np.argmax(predict[0])+65)
        nodetect = False
    
    # display the hand sign on window
    if nodetect == False:
        img = cv2.putText(canvas, char, (10, height+40), font, 1.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', img)
    else:
        cv2.imshow('frame', canvas)

cap.release()
cv2.destroyAllWindows()
