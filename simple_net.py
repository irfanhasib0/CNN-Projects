
# coding: utf-8

# In[2]:


'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from bs4 import BeautifulSoup
import cv2
import numpy as np
import glob
from PIL import Image
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt

i=0
nn=224
train_data=[]
print('abc')
for filename in glob.glob('images/train/*.jpg'):
    fname='Out/images/'+str(i)+'.jpg'
    image=cv2.imread(filename)
    image=cv2.resize(image,(nn,nn))
    train_data.append(image)
    #cv2.imwrite(fname,image)
    #print(i)
    i=i+1
    if(i==185):
        break
    
train_data=np.array(train_data)
#cv2.imshow('image',train_data[100])
#cv2.waitKey(0)
output=[]
i=0
print(train_data.shape)


for filename in glob.glob('annotations/train/*.xml'):
    f=open(filename)
    data=f.read()
    soup = BeautifulSoup(data, 'html.parser')

    height=int(soup.height.string)
    width=int(soup.width.string)

    xmin=int(soup.xmin.string)
    ymin=int(soup.ymin.string)
    xmax=int(soup.xmax.string)
    ymax=int(soup.ymax.string)

    h_fact=nn/height
    w_fact=nn/width

    xmin=int(xmin*w_fact)
    xmax=int(xmax*w_fact)
    ymin=int(ymin*h_fact)
    ymax=int(ymax*h_fact)    
    output.append([xmin,ymin,xmax,ymax])
    i=i+1
    if(i==185):
        break
train_label=np.asarray(output)
print(train_label.shape)


# In[3]:


batch_size = 10
num_classes = 4
epochs = 10
num_predictions = 20
model_name = 'indonasia.h5'

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print('x_train shape:', train_data.shape)

# Convert class vectors to binary class matrices.


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=train_data.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('relu'))
model.summary()


# In[ ]:


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_data = train_data.astype('float32')
train_data/=255
x_train=train_data[:61]
x_test=train_data[61:71]

y_train=train_label[:61]
y_test=train_label[61:71]


history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),shuffle=True)

model.save('simple_net.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
savefig('loss.jpg')
