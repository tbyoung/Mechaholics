#Data imports, general imports, manipulation imports
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
#Model Build Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

dataDir = 'data'
#Remove bad images
image_extensions = ['jpeg','jpg','bmp','png']
print(os.listdir(os.path.join(dataDir))) #TEST CODE
for image_class in os.listdir(dataDir):
    for image in os.listdir(os.path.join(dataDir, image_class)):
        image_path = os.path.join(dataDir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extensions:
                print('Image not in extension list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

#Data Load
data = tf.keras.utils.image_dataset_from_directory('data') #Build data pipeline. Default Parameters: Batch Size = 32, Image Size = (256,256), No validation split, bilinear interpolation
data_iterator = data.as_numpy_iterator() #Give us access to the dataset's data
batch = data_iterator.next() #obtain a 'batch' of the dataset (access the data pipeline)
#An element of 'batch' contains the image represented as an array, with 3 color channels for color images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#Preprocessing
#Scaling the colors (0-255 turning to 0-1)
data = data.map(lambda x, y: (x/255, y))
#Splitting
train_size = int(len(data)*.7) #70% training
val_size = int(len(data)*.2) #20% value
test_size = int(len(data)*.1) #10% testing
print("Data Length (len):" + str(len(data)))
print(" Val_size: "+str(val_size)+ " Train_size: " + str(train_size) + " Test_size: " + str(test_size))
#train+val+test should be equal to len(data)

#Allocating the data
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#CNN Model (Keras sequential API)
#Build
#model = Sequential([Conv2d()])
#A sequential model means that the data will flow sequentially through the layers, no deviation from layer pathing
#model.add adds layers to the neural network
#Each combination of Convolutional layer and pooling layer is a 'Convolutional Block'
model = Sequential()

#the first layer is the 'input layer' and must contain the input_shape parameter whose size is specified from our preprocessing
#These model parameters must be modified to increase performance of the model, further experimentation with a better dataset required for calibration
#16 filters that condense information, filter is 3 pixels by 3 pixels in size, with a 1 pixel stride (moves one pixel befor reapplying filter), 
#activation that gets applied is relu: takes output from convolutional layer and applies a function on the numerical output (relu converts all negative numbers to 0 then applies a linear transformation on values above 0)
#Activations are the reason that convolutional neural networks are powerful, without an activation it is just a linear neural network, which would be classified as just a deep neural network (less powerful than convolutional neural networks and negates any reason for convolutional layers)
#Sigmoid activation applies a function of (1/1+e^-x)
#A max pooling layer takes the maximum value after the (relu) activation and returns back that value, condensing the information in the process (Not just one number, but over regions of the image; default is 2x2 pixel regions)
#After applying a convolutional layer, the number of filters will form the channel value.
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

#32 filters, 3x3 pixel filter size, 1 stride, relu activation
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#16 filters, 3x3 pixel filter size, 1 stride, relu activation
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#Flattens the data, condenses the length and width of the data, condensing the layers to make 256 values
#Flatten converts from a multirank tensor to a single-rank tensor
model.add(Flatten())

#Dense layers determine the number of outputs to the next layer, making the data more "dense"
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#By condensing the layers to 1 output value, we get a situation by which (due to the sigmoid activation) the closer the value is to either 
#0 or 1, the the more likely it is to be of that classification (0 or 1; or in our case Okra or Nutgrass)

#Compilation
#Adam is the most common optomizer I saw on the internet, still undetermined as to why
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
#Model.summary() will display how the data looks and is altered as it passes through the neural network. The most important thing to pay attention to when getting a better understanding of what the layers are doing is the output shape
#Notice how the flatten layer multiplies the parameters of the pooling layer above it to determine the parameters of itselfm thereby "flattening" the data into one channel
model.summary()

#Train
logDir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logDir)

#two important functions when building a neural network, .fit() and .predict()
#.fit is the training component, whereas .predict is the testing/utilizing component
#Epochs is how long we train for, with 1 epoch being 1 run over the entire training dataset. 
#Validation data is used to evaluate the performance of the model, lets us know how the model is performing
#hist stands for history, and this will let us plot the performance of our model
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Plot
fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='blue', label='loss')
plt.plot(hist.history['val_accuracy'], color='red', label='val_loss')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper right')
plt.show()

#Performance
#Evaluation
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    x,y = batch
    yhat = model.predict(x)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)
print(f'Precision: {precision.result().numpy()}, Recall: {recall.result().numpy()}, Accuracy: {accuracy.result().numpy()}')
#Testing
img = cv2.imread('OkraTest.jpg')
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()
resize = tf.image.resize(img,(256,256))
#Model doesnt expect single images, it expects batches. np.expand_dims encapsulates the image, giving it another dimension and making it look like a batch
yhat = model.predict(np.expand_dims(resize/255, 0))
if(yhat<0.5):
    print('Classification: 0['+str(yhat)+']')
elif(yhat>0.5):
    print('Classification: 1['+str(yhat)+']')
else:
    print('Classification indeterminate (Classified as 0.5)')
img = cv2.imread('NutGrassTest.jpg')
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()
resize = tf.image.resize(img,(256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
if(yhat<0.5):
    print('Classification: 0['+str(yhat)+']')
elif(yhat>0.5):
    print('Classification: 1['+str(yhat)+']')
else:
    print('Classification indeterminate (Classified as 0.5)')

#SAVE & LOAD
#This will save the model in the models folder, make sure to check this folder and ensure it was updated properly
model.save(os.path.join('models', 'BinaryClassificationModel.tflite'))

#Test by loading
loadModel = load_model(os.path.join('models', 'BinaryClassificationModel.tflite'))

img = cv2.imread('OkraTest.jpg')
resize = tf.image.resize(img,(256,256))
yhat = loadModel.predict(np.expand_dims(resize/255, 0))
if(yhat<0.5):
    print('Classification: 0['+str(yhat)+']')
elif(yhat>0.5):
    print('Classification: 1['+str(yhat)+']')
else:
    print('Classification indeterminate (Classified as 0.5)')
img = cv2.imread('NutGrassTest.jpg')
resize = tf.image.resize(img,(256,256))
yhat = loadModel.predict(np.expand_dims(resize/255, 0))
if(yhat<0.5):
    print('Classification: 0['+str(yhat)+']')
elif(yhat>0.5):
    print('Classification: 1['+str(yhat)+']')
else:
    print('Classification indeterminate (Classified as 0.5)')