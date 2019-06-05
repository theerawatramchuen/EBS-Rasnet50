# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/dansbecker/transfer-learning

Created on Tue Jun  4 21:29:56 2019

@author: User
"""
import time
from keras.models import load_model

#Specify Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

num_classes = 2
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#Download at https://www.kaggle.com/keras/resnet50/downloads/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5/2

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

#Compile Model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Loading model weight
start = time.time()
# my_new_model.load_weights('EPB_rasnet50.h5')

# Fit Model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'dataset/training_set',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'dataset/test_set',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=562,
        validation_data=validation_generator,epochs = 50,
        validation_steps=140,
        callbacks=callbacks_list)

#my_new_model.save_weights('EBS_rasnet50_Eval.h5')

end = time.time()
print('Trainging time is',(end - start),' Seconds')