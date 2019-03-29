# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:04:01 2019

@author: krishna raj
"""

import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def prepare_image(file):

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

train_path = 'train'
valid_path = 'valid'
test_path = 'test'

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224,224), batch_size=20)

valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224,224), batch_size=20)

test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224,224), batch_size=20)
  

mobile = keras.applications.mobilenet.MobileNet()

x = mobile.layers[-6].output
predictions = Dense(3, activation= 'softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)

for layer in model.layers[:-5]:
    layer.trainable = False
    
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

step_size_train=train_batches.n//train_batches.batch_size
val_size_train=valid_batches.n//valid_batches.batch_size

model.fit_generator(train_batches, steps_per_epoch=125 , validation_data=valid_batches, validation_steps=21, epochs=1, verbose=2)
 

#predictions = model.predict_generator(test_batches, steps=1, verbose=2)
img_path = 'test.jpg'
preprocessed_image = prepare_image('test.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results
model.save('fine_tune_mobile.h5')
model.save_weights('fine_tune_mobile_weights.h5')
     