import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import matplotlib.pyplot as plt

TRAIN_DIR = 'D:\\Users\\Ivan\\Desktop\\FER\\Projekt Iz PP\\Projekti\\Cat,Dog\\train'
TEST_DIR = 'D:\\Users\\Ivan\\Desktop\\FER\\Projekt Iz PP\\Projekti\\Cat,Dog\\test'
IMG_SIZE = 50
LR = 1e-5 #Learning rate, How fast weights change

# just so we remember which saved model is which, sizes must match
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


#helper function to convert the image name to an array.
def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1, 0]
    #                            [no cat, very doggo]
    elif word_label == 'dog': return [0, 1]


'''
The  function converts the data for us into array
 data of the image and its label.

When we've gone through all of the images, 
we shuffle them, then save. 
Shuffle modifies a variable in place,
 so there's no need to re-define it here.

With this function, we will both save, and return the array data. 
This way, if we just change the neural network's structure, 
and not something with the images, like image size..etc..
then we can just load the array file and save some processing time.
'''

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):#tqdm=progress bar
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# if you need to create the data:
#train_data = create_train_data()

# If you have already created the dataset:
train_data = np.load('train_data.npy')

'''
Here we create a network with 7 layers.

First five layers are convolutional, and the last two are fully connected.
Every conv. layer has it own input and output.
Both input and output have their own (potentially different) sizes, in
the form of width*height*depth.

Input for the first conv. layer is a picture, if it's a colored picture,
the dimensions are width*height*depth (or R*G*B).

But in this example the picture is transformed into a black and white and
scaled to dimensions of IMG_SIZE*IMG_SIZE.
'None' in the first line means that there will be several, undefined number,
of pictures.

Convolution in 2D is basically summ of weights for each pixel and pixels around
it( 5x5 for example).
Every value is multiplied by some weight value(those factors are trained by
the network). That is called convolutional filter.

The result of calculating summ of each pixel is a 2d matrix similar to the
original picture called feature map.
The resulting image is the same size as the original image, or slightly smaller
due to the pixels on the edge not having 5*5 pixels around them.

One convolutional layer can have, and mostly has, more convolutional filters,
and every one of them produces it's own feature map.
So if the layer has 32 convolutional filers, and the area for summ calculation
is 5x5, the output of the whole layer will be 5*5*32.                                             
'''

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#make a new conv. layer with the dimensions of the previous one,
#with 32 filters, and the dimensions of filers are 5*5
convnet = conv_2d(convnet, 32, 5, activation='relu')#convolutional = convnet
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')#output, dog or cat
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# if you need to create the data:
#test_data = process_test_data()

# if you already have some saved:
test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()