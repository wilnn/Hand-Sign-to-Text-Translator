###############

# this file is to test the trained model

###############

import os
import numpy as np
import tensorflow as tf
import keras
from dotenv import load_dotenv
import cv2

def test_model(model):
    currentpath = os.getcwd()
    featuresFiles = os.listdir(currentpath + "\\processed_data\\data_for_testing")
    test_ds = np.load(currentpath + "\\processed_data\\data_for_testing\\" + featuresFiles[0])
    print(test_ds.shape)
    labels_ds = np.full(test_ds.shape[0], ord(featuresFiles[0][0])%65)

    for i, file in enumerate(featuresFiles):
        if file == featuresFiles[0]:
            continue
        test = np.load(currentpath + "\\processed_data\\data_for_testing\\" + file)
        if file == 'del.npy':
            label = np.full(test.shape[0], 26)
        elif file == 'space.npy':
            label = np.full(test.shape[0], 27)
        else:
            label = np.full(test.shape[0], ord(file[0])%65)
        # label = np.full(test.shape[0], ord(file[0])%65)
        test_ds = np.append(test_ds, test, axis=0)
        labels_ds = np.append(labels_ds, label, axis=0)

    print(test_ds.shape)
    print(labels_ds.shape)
    randomize = np.arange(len(test_ds))
    np.random.shuffle(randomize)
    test_ds = test_ds[randomize]
    labels_ds = labels_ds[randomize]

    #print(test_ds[0].shape)
    test_dss = test_ds/255.0

    # dectivate dropout layer for better evaluation result
    for layer in model.layers:
        if type(layer) == keras.src.layers.regularization.dropout.Dropout:
            layer.training = False

    loss, acc = model.evaluate(test_dss, labels_ds, verbose=1)

    # predic the first few images in the test dataset
    for i, f in enumerate(test_ds):
        img = (np.expand_dims(f,0))
        predict = model.predict(img/255.0)
        print(np.argmax(predict[0]))
        print('real result:', labels_ds[i])
        cv2.imshow('d', f)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    return loss, acc

def main():
    model = tf.keras.models.load_model('models/hand_sign_translator.keras')
    print(model.summary())
    loss, acc = test_model(model)
    print(f'Accuracy: {acc},    Loss: {loss}')

if __name__ == "__main__":
    main()