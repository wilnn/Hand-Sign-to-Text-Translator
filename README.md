# Hand Sign to Text Translator
![Static Badge](https://img.shields.io/badge/python-3.9%20%7C%203.12-blue?labelColor=gray)
<!-- table of contents-->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About-The-Project">About The Project</a>
      <ul>
      <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#How-It-Works">How It Works</a></li>
    <li><a href="#Examples/Demonstration">Examples/Demonstration</a></li>
    <li><a href="#Repository-details">Repository details</a></li>
    <li><a href="#Possible-Improvements">Possible Improvements</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project
* A program that can translate the ASL (American Sign Language) alphabet to text in real-time via camera using Tensorflow, OpenCV, and Mediapipe, can be used to write English sentences with just hand signs.
* A convolutional neural network (CNN) was created and trained using TensorFlow. The model was trained with about 70,000 images of 28 different hand signs. It was tested with 1,400 images and has the **accuracy of 99.09%**.

### Built With
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-black?labelColor=orange)][TensorFlow-url]
* [![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.84-black?labelColor=green)][OpenCV-url]
* [![Mediapipe](https://img.shields.io/badge/Mediapipe-4.10.0.84-black?labelColor=blue)][Mediapipe-url]
* [![Numpy](https://img.shields.io/badge/Numpy-1.26.4-black?labelColor=yellow)][Numpy-url]
    * <u>Note: OpenCV version 4.10.0.84 only works with Numpy version 1.26.4. Using Numpy version 2.0 will cause an error.</u>

## How It Works
* The pre-trained Google's Mediapipe hand landmark model is used to detect the hand in images obtained from the camera. Then, the hand is passed to the CNN model that I trained to get an output that is a number from 0-27, with 0-25 representing the corresponding alphabet letters, 26 means deleting the previous letter, and 27 means adding space to write the next words.

## Examples/Demonstration
![image](https://github.com/user-attachments/assets/65565ffe-99b3-40e4-9d37-ae966bea8e1d)

https://github.com/user-attachments/assets/c3b8eac1-067c-42ba-823c-b66d034a1982
<!--THE BLANK LINE BETWEEN THE VIDEO LINK AND THE IMAGE IS NEEDED FOR THE VIDEO TO LOAD-->
## Repository Details
### Code
* **_init_.py**: contains the code for the main Hand Sign to Text Translator program.
* **create_input.py**: contains the code to create the .npy files in the **processed_data** directory. 
* **train_model.py**: this file is used to create and train the CNN model using the npy files in the **processed_data** directory as inputs. The trained model is saved inside the **models** directory as **hand_sign_translator.keras**
* **test.py**: used to test the model **hand_sign_translator.keras** that is trained and saved in the model's directory.
### models
* **hand_landmarker.task**: Google's Mediapipe hand landmarker model that is used for hand recognition.
* **hand_sign_translator.keras**: the model that was trained using train_model.py
### processed_data
* Each file inside was created by the **create_input.py** file. These files contain images of the hands that are represented as Numpy arrays.

## Possible Improvements
* The trained model can not recognize the hand sign for letters U, T, and N as well as the others due to the similarity between hand signs and the quality of the training dataset. Better results can be achieved by having a more varied dataset and increasing the model depth.
* Overall, ASL hand sign recognition is a challenging task because people have different hand sizes, shapes, colors, and slightly different ways to do the same hand sign. The more variety of the training dataset, the better the model will be. 

## License
Distributed under the MIT License. See LICENSE.txt for more information.

## Contact
William Nguyen - thangnguyen15700@gmail.com

## Acknowledgments
* The dataset is obtained from Kaggle: [https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)

<!-- link-->
[TensorFlow-url]: https://www.tensorflow.org/
[OpenCV-url]: https://opencv.org/
[Mediapipe-url]: https://pypi.org/project/mediapipe/
[Numpy-url]: https://pypi.org/project/numpy/
