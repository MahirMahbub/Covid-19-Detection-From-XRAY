from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import pickle


class PredictorConfig(AppConfig):
    modelFile = './predApp/model/COVID_XRAY_nor.h5'  
    model = load_model(modelFile)


    