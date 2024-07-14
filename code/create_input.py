import cv2
import os
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
import math

def create_input_file(name, path):
    currentpath = os.getcwd()
    featuresdir = os.listdir(currentpath + path)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # create a hand landmarker instance with the image mode
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.getenv('modelPath')),
    running_mode=VisionRunningMode.IMAGE)

    # loop through each folder in the parent folder
    for dir in featuresdir:
        # loop through each iamge file in each alphabet folder in the parent folder
        files = os.listdir(currentpath + path + '\\' + dir)
        # need to create a fixed size numpy first because it will be faster than appending
        featuresArray = np.zeros((len(files), 190, 190, 3)) # if want for grayscale iamge then remove 3
        # labelsArray = np.empty(len(files), dtype=str)
        n = 0
        for file in files:
            img = cv2.imread(currentpath + path + '\\' + dir + '\\' + file, cv2.IMREAD_COLOR) # load in an imamge in BGR format as default in opencv
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB format

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            with HandLandmarker.create_from_options(options) as landmarker:
                landmarker_result = landmarker.detect(mp_image)

            if not landmarker_result or not landmarker_result.hand_landmarks:
                featuresArray = featuresArray[:-1]
                labelsArray = labelsArray[:-1]
                continue
            # get the coner for the bounding box of the hand by finding
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

            # convert to pixel
            min_y = round((min_y -0.017) * img.shape[0])
            max_y = round((max_y + 0.017) * img.shape[0])
            min_x = round((min_x - 0.017) * img.shape[1])
            max_x = round((max_x + 0.017) * img.shape[1])
            
            hand = img[min_y:max_y, min_x:max_x]
            hand = cv2.resize(hand, (190,190))
            print(hand.shape)
            cv2.imshow('Image', hand)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # print(landmarker_result.hand_landmarks[0][0].x)
            featuresArray[n] = hand
            # labelsArray[n] = dir
            n+=1
            # time.sleep(100000000)


            # will resize hand to 250x250
            # creaate npy file for each alphabet folder.
        if path == os.getenv('trainPath'):
            np.save(currentpath + '\\processed_data' + '\\data_for_training' + '\\' + dir + '.npy', featuresArray)
        elif path == os.getenv('testPath'):
            np.save(currentpath + '\\processed_data' + '\\data_for_testing' + '\\' + dir + '.npy', featuresArray)
        else:
            np.save(currentpath + '\\processed_data' + '\\data_for_validation' + '\\' + dir + '.npy', featuresArray)

def main():
    load_dotenv()
    create_input_file('training', os.getenv('trainPath'))
    create_input_file('testing', os.getenv('testPath'))
    create_input_file('testing', os.getenv('validationPath'))

if __name__ == "__main__":
    main()

