# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:01:11 2019

@author: krishna raj
"""

import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.utils.generic_utils import CustomObjectScope





def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):    
    new_model = load_model('fine_tune_mobile.h5')

preprocessed_image = prepare_image('test.jpg')
predictions = new_model.predict(preprocessed_image)
#results = imagenet_utils.decode_predictions(predictions)
#results
print(predictions)
