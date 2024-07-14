import cv2
import os
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import time

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
        featuresArray = np.zeros((len(files), 250, 250, 3)) # if want for grayscale iamge then remove 3
        labelsArray = np.zeros(len(files))
        for file in files:
            img = cv2.imread(currentpath + path + '\\' + dir + '\\' + file, cv2.IMREAD_COLOR) # load in an imamge in BGR format as default in opencv
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB format

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            with HandLandmarker.create_from_options(options) as landmarker:
                landmarker_result = landmarker.detect(mp_image)

            if not landmarker_result:
                featuresArray = featuresArray[:-1]
                labelsArray = labelsArray[:-1]
                continue

            print(landmarker_result.landmarks)
            time.sleep(100000000)


            # will resize hand to 250x250
            # creaate npy file for each alphabet folder.




def main():
    load_dotenv()
    create_input_file('training', os.getenv('trainPath'))
    create_input_file('testing', os.getenv('testPath'))
    create_input_file('testing', os.getenv('validationPath'))

if __name__ == "__main__":
    main()

