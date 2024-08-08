#######

# this file is used to train the model

#######
import os
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import cv2

def create_model(image_shape, num_classes):
# create the structure of the model
    # first layer always need to specify input shape. last layer specify the number of different labels that the data can be classify into
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=image_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    #tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu'),
    #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])

    # create model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, epochs):

    currentpath = os.getcwd()
    featuresFiles = os.listdir(currentpath + os.getenv('processedTrainData'))
        
    # get the smallest .npy file
    smallestsize = os.path.getsize(currentpath + os.getenv('processedTrainData') + '\\' + featuresFiles[0])
    smallestfile = featuresFiles[0]
    for file in featuresFiles:
        size = os.path.getsize(currentpath + os.getenv('processedTrainData') + '\\' + file)
        if size < smallestsize:
            smallestsize = size
            smallestfile = file
    
    smallestDS = np.load(currentpath + os.getenv('processedTrainData') + '\\' + smallestfile)

    ######
    # uncoment line 1 below and comment line 2 if
    # the dataset is too big and can not fit them all into memory.
    ######
    #length = smallestDS.shape[0] # this is line 1
    length = 1 # this is line 2
    index1 = 0
    index2 = 650

    # loop to train the model with small segment of numpy array of all the .npy file in the folder
    while length > 0:
        
        memmap_array = np.load(currentpath + os.getenv('processedTrainData') + '\\' + smallestfile, mmap_mode='r')
        if length > 650:
                # train_ds = np.empty([650, 190, 190])
                train_ds = memmap_array[index1:index2]
        else:
            #train_ds = np.empty([length, 190, 190])
            train_ds = memmap_array[index1:]
        if smallestfile == 'del.npy':
            labels_ds = np.full(train_ds.shape[0], 26)
        elif smallestfile == 'space.npy':
            labels_ds = np.full(train_ds.shape[0], 27)
        else:
            labels_ds = np.full(train_ds.shape[0], ord(smallestfile[0])%65)

        # loop through each npy file to get the small segment of each files and append each small segment into 1 big array
        for file in featuresFiles:
            if file == smallestfile:
                continue  
            # get the training data from .npy file and create labels
            memmap_array = np.load(currentpath + os.getenv('processedTrainData') + '\\' + file, mmap_mode='r')
            if length > 650:
                partial = memmap_array[index1:index2]
            else:
                partial = memmap_array[index1:]
            
            if file == 'del.npy':
                partial_labels = np.full(partial.shape[0], 26)
            elif file == 'space.npy':
                partial_labels = np.full(partial.shape[0], 27)
            else:
                partial_labels = np.full(partial.shape[0], ord(file[0])%65)
            
            train_ds = np.append(train_ds, partial, axis=0)
            labels_ds = np.append(labels_ds, partial_labels, axis = 0)

        randomize = np.arange(len(train_ds))
        np.random.shuffle(randomize)
        train_ds = train_ds[randomize]
        labels_ds = labels_ds[randomize]

        train_ds = train_ds[:, :, :, np.newaxis]

        train_ds = train_ds/255.0
        history = model.fit(train_ds, labels_ds, validation_split=0.2, epochs=epochs)
        print(history.history)
        length -= 650
        index1 += 650
        index2 += 650

    return model

def main():
    load_dotenv()
    model = create_model((190, 190, 1), 28)
    model = train_model(model, 25)
    model.save("hand_sign_translator.keras")

if __name__ == "__main__":
    main()
