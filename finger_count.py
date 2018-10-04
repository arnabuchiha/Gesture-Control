from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten,Input,GlobalAveragePooling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,load_model
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import tensorflow as tf
import numpy as np
import argparse
from keras import backend as K
from tensorflow.python.client import device_lib
import os
tf.__version__
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

IM_WIDTH, IM_HEIGHT = 300,300
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024

def setup_transfer_learning(model,base_model):
    base_model.trainable = False
    
    model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.SGD(lr=1e-4, 
                                       momentum=0.9),
              metrics=['accuracy'])

def last_layer(base_model,nb_classes):
    x=Sequential()
    x.add(base_model)
    x.add(GlobalAveragePooling2D())
    x.add(Dropout(0.3))
    x.add(Dense(6, activation='softmax'))
    x.summary()
    return x

def train():
    nbatch=32

    train_datagen=ImageDataGenerator(   rescale=1/.255,
                                    rotation_range=10.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True
                                    )
    test_datagen=ImageDataGenerator(rescale=1/.255)

    train_gen=train_datagen.flow_from_directory('D:/datasets/finger_count/images/train/',
                                           target_size=(300,300),
                                           color_mode='grayscale',
                                           batch_size=nbatch,
                                           classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                           class_mode='categorical'
                                        )

    test_gen=test_datagen.flow_from_directory('D:/datasets/finger_count/images/test/',
                                           target_size=(300,300),
                                           color_mode='grayscale',
                                           batch_size=nbatch,
                                           classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                           class_mode='categorical'
                                           )
        
        
    # base_model=InceptionV3(weights='imagenet',include_top=False,input_shape=(300,300,3))
    
    # model=last_layer(base_model,6)
    # setup_transfer_learning(model,base_model)
    
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(300,300,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))

    model.summary()
    opt = optimizers.SGD(lr=0.01)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    
    callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath='model_6cat.h5', monitor='val_loss', save_best_only=True),
    ]
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)

    with tf.device('/device:GPU:0'):
        history = model.fit_generator(
        train_gen,
        steps_per_epoch=71,
        epochs=40,
        validation_data=test_gen,
        validation_steps=28,
        callbacks=callbacks_list
        )
        # session.run(history)
        print("Reached")
    plot_training(history)
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')
    
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('loss.png')

print(device_lib.list_local_devices())
train()