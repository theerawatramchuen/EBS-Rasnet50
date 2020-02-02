# -*- coding: utf-8 -*-
"""

Created on Tue Jun  4 21:29:56 2019

@author: Theerawat Ramchuen

Update 2/2/2020

"""
# Training Parameter ###########################################
TRAINING_PATH = 'D:/EBS-Rasnet50/dataset/training_set'
VALIDATION_PATH = 'D:/EBS-Rasnet50/dataset/test_set'
BATCH_SIZE = 16
LEARNING_RATE = 2.5e-05
EPOCH = 100
################################################################

# Code Start here
import tensorflow
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

#Specify Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#Get Number of Class 
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

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg',))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say yes to train first layer (ResNet) model.
my_new_model.layers[0].trainable = True

# Optimizer
sgd = tensorflow.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)

#Compile Model
my_new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.summary()

# checkpoint
filepath='checkpoint/{epoch:03d}-acc_{acc:.5f}-valacc_{val_acc:.5f}-valloss{val_loss:.3E}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                save_best_only=False, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                patience=10, verbose=1, mode='auto', 
                                min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint,reduce_lr]


start = time.time()
# Loading model pretrain weight
#my_new_model.load_weights('checkpoint/017-acc_1.00000-valacc_1.00000.hdf5')

# Fit Model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        TRAINING_PATH,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        VALIDATION_PATH,
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(train_generator,
        validation_data=validation_generator,
        epochs = EPOCH,
        callbacks=callbacks_list)

end = time.time()
print('Trainging time is',(end - start),' Seconds')