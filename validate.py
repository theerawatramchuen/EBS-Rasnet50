# -*- coding: utf-8 -*-
"""

Created on Tue Jun  4 21:29:56 2019

@author: Theerawat Ramchuen

Update 2/2/2020

"""
## Validation Parameter ###################################################
WEIGHT = '017-acc_1.00000-valacc_1.00000.hdf5'
TRAINING_PATH = 'D:/EBS-Rasnet50/dataset/training_set'
TEST_IMAGE = 'dataset/single_prediction/sample.jpg'
###########################################################################

# Code Start Here
WEIGHT = 'checkpoint/' + WEIGHT
# Importing the Keras libraries and packages
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense

# Initialise the number of classes
# Get Number of Class 
import os
def fcount(path, map = {}):
  count = 0
  for f in os.listdir(path):
    child = os.path.join(path, f)
    if os.path.isdir(child):
      child_count = fcount(child, map)
      count += child_count + 1 # unless include self
  map[path] = count
  return count
map = {}
num_classes = fcount(TRAINING_PATH, map)
#num_classes = 2
 
# Build the model
classifier = Sequential()
classifier.add(ResNet50(include_top=False, pooling='avg'))
classifier.add(Dense(num_classes, activation='softmax'))
 
# Say yes to train first layer (ResNet) model.
classifier.layers[0].trainable = True
 
# Compiling the CNN
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Loading model weight
start = time.time()
classifier.load_weights(WEIGHT)

#Prediction Image filename cat_or_dog.jpg
from keras.preprocessing import image as image_utils
test_image = image_utils.load_img(TEST_IMAGE, target_size = (224, 224))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

print ('Good ',result[0][0]*100.0,'%')
print ('Reject ',result[0][1]*100.0,'%')

end = time.time()
print('Prediction time is',(end - start),' Seconds')
print(result.shape)