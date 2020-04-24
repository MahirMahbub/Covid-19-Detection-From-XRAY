# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:48:16 2020

@author: Mahir Mahbub
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
from shutil import copyfile
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image


INIT_LR = 1e-3
EPOCHS = 50
BS = 8
# %%

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224, 3))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    #img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# %%
X = []
Y = []
import glob
Covic_files = glob.glob(r"D:\covid-chestxray-dataset-master\training\Covic"+"\*")
for fil in Covic_files:
    img = load_image(fil)
    X.append(img)
    Y.append([1, 0])

Non_covic_files = glob.glob(r"D:\covid-chestxray-dataset-master\training\Non_covic"+"\*")
for fil in Non_covic_files:
    img = load_image(fil)
    X.append(img)
    Y.append([0, 1])

# %%    
#weights="imagenet",   

#
def model_create(): 
    baseModel = InceptionResNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
    	layer.trainable = True
    opt = Adam(lr=INIT_LR, decay=INIT_LR / (EPOCHS))
    model.compile(loss=tf.keras.losses.CosineSimilarity(axis=1), optimizer="sgd",
	metrics=["accuracy"])
    return model

'''
train_generator = train_datagen.flow_from_directory(
        path ,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        path_on+"/test" ,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='categorical')
'''


# %%
model = 0
n_split=10
X = np.array(X)
Y = np.array(Y)
Y = Y.astype(np.float64)
X = X.astype(np.float64)
count = 1
accuracy = 0
for train_index,test_index in KFold(n_split, random_state=77, shuffle=True).split(X):
    trainX, testX=X[train_index],X[test_index]
    trainY, testY=Y[train_index],Y[test_index]
    model = model_create()
    
    count+=1
    #brightness_range=[0.9,1.2], 
    trainAug = ImageDataGenerator(zoom_range = [1, 1.15])
    testAug = ImageDataGenerator()
    es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=20)
    
    H = model.fit_generator(
    	trainAug.flow(trainX, trainY, batch_size=BS) ,
    	#steps_per_epoch= (217 // BS)+1,
        validation_data=testAug.flow(testX, testY, batch_size=BS),
        #validation_steps= (25// BS)+1,
    	epochs=EPOCHS,
        callbacks=[es])
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testY.argmax(axis=1), predIdxs,target_names=["Covic", "Non_Covic"]))
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    accuracy += (acc/10)
    # show the confusion matrix, accuracy, sensitivity, and specificity
    with open('out.txt', 'a') as f:
        print(classification_report(testY.argmax(axis=1), predIdxs,target_names=["Covic", "Non_Covic"]), file = f)
        #print('Filename:', filename, file=f) 
        print(cm)
        print()
        print("acc: {:.4f}".format(acc), file=f)
        print()
        print("sensitivity: {:.4f}".format(sensitivity), file=f)
        print()
        print("specificity: {:.4f}".format(specificity), file=f)
        print()
    model.save("COVID_XRAY_nor_"+str(count)+".h5")
#rotation_range=15,
#fill_mode="nearest"


