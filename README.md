# Hand Sign to Text Translator

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About-The-Project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project
* A program that can translate the ASL (American Sign Language) alphabet to text in real time via camera using Tensorflow, OpenCV, and Mediapipe, and can be used to write any english sentences with just hand signs.
* A convolutional neural network (CNN) was created and trained using TensorFlow. The model was trained with about 70,000 images of 28 different handsigns. It was tested with 1,400 images and has the accuracy of 99.09%.

### built with
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-black?labelColor=orange)][TensorFlow-url]
* [!OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.84-black?labelColor=green)][OpenCV-url]
* ![Mediapipe](https://img.shields.io/badge/Mediapipe-4.10.0.84-black?labelColor=blue)[Mediapipe-url]

## How It Works
* The pre-trained Google's Mediapipe hand landmarker model is used to detect the hand in images obtained from the camera. Then, the hand is passed to the CNN model that I trained to get an output that is a number from 0-27, with 0-25 is the represent the coresspoding alphabet letters, 26 means deleting the previous letter, and 27 means adding space to write the next words.

## Examples/Demonstration
![ASL alphabet](image.png)
<video width="320" height="240" controls>
  <source src="video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Files Breakdown
### Code
* _init_.py: contains the code for the main handsign to text translator program.
* create_input.py: contains the code to create the .npy files in the processed_data directory. 
* train_model.py: this file is used to create and train the CNN model using the npy files in the processed_data directory as inputs. The trained model is saved inside the models folder as 'hand_sign_translator.keras'
* test.py: uses to test the model 'hand_sign_translator.keras' that is trained and saved in the models directory.
### models
* hand_landmarker.task: Google's Mediapipe hand landmarker model that is used for hand recognition.
* hand_sign_translator.keras: the model that was trained using train_model.py
### images
* Each directory inside contain the images coresspond to the handsign.
### processed_data
* Each files inside was created by create_input.py file. These file contain the images of the hands that is represented as numpy arrays.

## Room for Improvements
* Overall, ASL hand sign recognition is a challenging task because people have different hand size, shape, color, and slightly different ways to do the same hand sign. The more variety of the training dataset, the better the model will be. 

## Author/contacts


## Acknowledgments
* The dataset is obtain from Kaggle: [https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)


<!-- link-->
[TensorFlow-url]: https://www.tensorflow.org/
[OpenCV-url]: https://opencv.org/
[Mediapipe-url]: https://pypi.org/project/mediapipe/