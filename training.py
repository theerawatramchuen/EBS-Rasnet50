# -*- coding: utf-8 -*-
"""

Created on Tue Jun  4 21:29:56 2019

@author: User
"""
import tensorflow
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers

#Specify Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

num_classes = 2

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg',))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say yes to train first layer (ResNet) model.
my_new_model.layers[0].trainable = True

# Optimizer
sgd = tensorflow.keras.optimizers.Adam(lr=2.5e-04, beta_1=0.9, beta_2=0.999, amsgrad=False)

#Compile Model
my_new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.summary()

# checkpoint
filepath='{epoch:03d}-acc_{acc:.5f}-valacc_{val_acc:.5f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Loading model weight
start = time.time()
#my_new_model.load_weights('050-acc_1.00000-valacc_1.00000.hdf5')

# Fit Model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'D:/EBS-Rasnet50/dataset/training_set',
        target_size=(image_size, image_size),
        batch_size=16,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'D:/EBS-Rasnet50/dataset/test_set',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        validation_data=validation_generator,epochs = 100,
        callbacks=callbacks_list)

end = time.time()
print('Trainging time is',(end - start),' Seconds')