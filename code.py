import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

Train_Dir = r"C:\Users\shahe\Desktop\digits_Train"
Test_Dir = r"C:\Users\shahe\Desktop\digits_Test"
img_size = 50
lr = 1e-3

model_name = "sign language digits-{}-{}".format(lr, '10convSignLang')


def label_img(img):
    word_label = img.split()[0]
    if word_label == 'zero':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif word_label == 'one':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif word_label == 'two':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif word_label == 'three':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif word_label == 'four':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif word_label == 'five':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif word_label == 'six':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif word_label == 'seven':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif word_label == 'eight':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif word_label == 'nine':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(Train_Dir)):
        label = label_img(img)
        path = os.path.join(Train_Dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('training_digits_sign.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_Dir)):
        path = os.path.join(Test_Dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_digits_sign.npy', testing_data)
    return testing_data


train_data = create_train_data()



import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
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

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print("model loaded")

train = train_data[:-50]
test = train_data[-50:]

X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,img_size, img_size,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=30, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=model_name)

model.save(model_name)

import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')
"""
fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 3, num + 1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'zero'
    elif np.argmax(model_out) == 1:
        str_label = 'one'
    elif np.argmax(model_out) == 2:
        str_label = 'two'
    elif np.argmax(model_out) == 3:
        str_label = 'three'
    elif np.argmax(model_out) == 4:
        str_label = 'four'
    elif np.argmax(model_out) == 5:
        str_label = 'five'
    elif np.argmax(model_out) == 6:
        str_label = 'six'
    elif np.argmax(model_out) == 7:
        str_label = 'seven'
    elif np.argmax(model_out) == 8:
        str_label = 'eight'
    elif np.argmax(model_out) == 9:
        str_label = 'nine'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
"""


