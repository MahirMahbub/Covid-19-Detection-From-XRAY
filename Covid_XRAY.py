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


INIT_LR = 1e-3
EPOCHS = 50
BS = 8

meta_data = pd.read_csv(r"C:\Users\Mahir Mahbub\Desktop\covid-chestxray-dataset-master\metadata.csv")
meta_data_needed = meta_data[['finding', 'filename']]

covid_file_names = meta_data_needed[meta_data_needed['finding']=="COVID-19"]
non_covid_file_names = meta_data_needed[meta_data_needed['finding']!="COVID-19"]

covid_file_names_only = covid_file_names["filename"].values
non_covid_file_names_only = non_covid_file_names["filename"].values

path_on = os.getcwd()
path = os.getcwd()+"/training"
flag = 0
try:
    os.makedirs(path+"/Covic")
except :
    print("Already Created")
    flag = 1
    

covid_path = path+"/Covic"

try:
    os.makedirs(path+"/Non_covic")
except:
    print("Already Created")

non_covid_path = path+"/Non_covic"

src = path_on+"/images"
if flag ==0:

    for file in covid_file_names_only:
        copyfile(src+"/"+file, covid_path+"/"+file)
        
    for file in non_covid_file_names_only:
        copyfile(src+"/"+file, non_covid_path+"/"+file)
# %%    
#weights="imagenet",     
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
  
train_datagen = ImageDataGenerator(
        rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=tf.keras.losses.CosineSimilarity(axis=1), optimizer="sgd",
	metrics=["accuracy"])



H = model.fit_generator(
	train_generator ,
	steps_per_epoch=116 // BS,
    validation_data=validation_generator,
    validation_steps=30// BS,
	epochs=EPOCHS,callbacks=[es])

# %%
model.save("COVID_XRAY.h5")

# %%

from tensorflow.keras.preprocessing import image

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224, 3))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

import glob
files = glob.glob(path_on+"/test/Covic/"+"/*")
print("From Covid Dataset.............")
for fil in files:
    img = load_image(fil)
    val = model.predict(img)
    if val[0][0]> val[0][1]:
        print("Covid")
    else:
        print("Non Covid")
    
    
print("From Non Covid Dataset............")
n_files = glob.glob(path_on+"/test/Non_covic/"+"/*")
print()
for fil in n_files:
    img = load_image(fil)
    val = model.predict(img)
    if val[0][0]< val[0][1]:
        print("Non Covid")
    else:
        print("Covid")
        
# %%

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# %%
open("converted_model_covid_xray.tflite", "wb").write(tflite_model)