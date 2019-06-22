strpath = "dataset/test_set/goods/"
threshold = 0.9

import numpy as np
import time
import cv2 as cv
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as image_utils

# Initialise the number of classes
num_classes = 2
 
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
classifier.load_weights('047-acc_1.00000-valacc_1.00000.hdf5')

print("path image : "+strpath)

list_imgfile = []

for root, directory, file in os.walk(strpath):
          for file_selected in file:
                    if '.jpg' in file_selected:
                              list_imgfile.append(root+file_selected)

cv.namedWindow('imageRun',cv.WINDOW_FREERATIO)
img = cv.imread(list_imgfile[0],1)
cv.imshow('imageRun',img)
i = 0
qty_good = 0
qty_reject = 0
for f in list_imgfile:
          i=i+1
          start = time.time()
          img = cv.imread(f,1)
          test_image = image_utils.load_img(f, target_size = (224, 224))
          test_image = image_utils.img_to_array(test_image)
          test_image = np.expand_dims(test_image, axis = 0)
          result = classifier.predict(test_image)
          prediction = 'REJECT'
          score = result[0][1]
          if  score < threshold :
                    prediction = 'REJECT'
                    qty_reject = qty_reject + 1
          else:
                    prediction = 'GOOD'
                    qty_good = qty_good + 1 
          prediction = prediction + " Score: " + str(round(score*100))
          img = cv.putText(img,prediction,(10,20),cv.FONT_HERSHEY_SIMPLEX,0.5,(25,255,0),1)
          cv.imshow('imageRun',img)
          cv.moveWindow('imageRun', 20,20)
          end = time.time()
          print("Process : ",str(i)," ",prediction," ",score," ",round(1000*(end-start),1),"mS")
          
          cv.waitKey(1000)
          del(img)

cv.waitKey(1000)
cv.destroyAllWindows()
print ("Qty Good : ", qty_good, "ea.", round((qty_good/(qty_good+qty_reject))*100,2),"%")
print ("Qty Rej  : ", qty_reject, "ea.", round((qty_reject/(qty_good+qty_reject))*100,2),"%") 
print ('Qty TTL  : ', qty_good+qty_reject)
