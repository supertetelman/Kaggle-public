from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model, model_from_json

from keras.layers import Input,  merge, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Dense, Flatten, Highway, Reshape, Dropout, LSTM, Merge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,\
    EarlyStopping
from keras import backend as K

from data import load_train_data, load_test_data

config = {}
# actual images is 580 (width) by 420 (height) (83 x 60 is 7x smaller
config['IMG_ROWS'] =32
config['IMG_COLS'] = 64

config['EPOCH'] = 200
config['BATCH'] = 16
config['VALIDATION'] = .25

config['BLUR'] = False #This doesn't seem to help much with noise in the model
config['RESIZE'] = True

if not config['RESIZE']:
    config['IMG_ROWS'] = 420
    config['IMG_COLS'] = 580
    
config['IMG_COUNT'] = 222222 # Number of images to use
if config['IMG_COUNT'] > 5635:
    config['IMG_COUNT'] = 5635
    

config['LOAD_LAST'] = False # Load the last recorded weights
config['LOAD_MODEL'] = False # Load the last model file rather than making a new one 
config['SMOOTH'] = 1 # Prevent dice coef to be undefined when no mask is found
config['FIT'] = True # Fit the data or not, skip image pre-process if False

config['lr'] = 1e-5 # Learning rate used by optimizer

    
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config['SMOOTH']) / (K.sum(y_true_f) + K.sum(y_pred_f) + config['SMOOTH'])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    '''Play around with various convolutions, noise, etc. here'''
    inputs = Input((1, config['IMG_ROWS'], config['IMG_COLS']))

#    a = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(inputs)
#    a = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(a)
    
#    b = Convolution2D(12, 3, 3, activation='relu', border_mode='same')(a)
#    b = Convolution2D(12, 3, 3, activation='relu', border_mode='same')(b)
    #inputs = Dropout(.8)(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1b = Dropout(1)(pool1)
    pool1 = Dropout(.5)(pool1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(1)(pool2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(.5)(pool3)


    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(.5)(pool4)


    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = Dropout(.5)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = Dropout(.5)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = Dropout(.5)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = Dropout(.5)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = Dropout(.5)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    
#    conv10 = Convolution2D(1, 3, 3, activation='relu')(conv9)
#    conv10 = MaxPooling2D((2,2), strides=(2,2))(conv10)

#    x = Flatten()(conv9)
#    h = Highway()(x)
    
#    conv10 = conv10.flatten(ndim=2)
#    d = Dense(32, activation='relu')(h)
#    d = Dense(8, activation='sigmoid')(d)
#    d = Dense(1, activation='softmax')(d)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=config['lr']), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs, all=False):
    if all:
        config['IMG_COUNT'] = imgs.shape[0]
    imgs_p = np.ndarray((config['IMG_COUNT'], imgs.shape[1], config['IMG_ROWS'], config['IMG_COLS']), dtype=np.uint8)
    for i in range(config['IMG_COUNT']):
        if config['RESIZE']:
            imgs_p[i, 0] = cv2.resize(imgs[i, 0], (config['IMG_COLS'], config['IMG_ROWS']), interpolation=cv2.INTER_AREA )
        else:
            imgs_p[i,0] = imgs[i,0]
        if config['BLUR']:
            imgs_p[i, 0] = cv2.GaussianBlur(imgs_p[i, 0], (2, 2),100) # was 3,3,150
    return imgs_p


def pp_head(string):
        print('-'*30)
        print(string)
        print('-'*30)    


def train_and_predict():
    stats = {}
    pp_head(config)
    open('config.txt', 'w').write(str(config))
    
    if  True or config['FIT']:
        pp_head('Loading and preprocessing train data...')
        imgs_train, imgs_mask_train = load_train_data()
    
        imgs_train = preprocess(imgs_train)
        imgs_mask_train = preprocess(imgs_mask_train)
    
        imgs_train = imgs_train.astype('float32')
        stats['mean'] = np.mean(imgs_train)  # mean for data centering
        stats['std'] = np.std(imgs_train)  # std for data normalization
    
        imgs_train -= stats['mean']
        imgs_train /= stats['std']
        
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_mask_train /= 255.  # scale masks to [0, 1]
        
        open('stats.txt', 'w').write(str(stats))
    else:
        stats = eval(open('stats.txt', 'r').read()) # Read previously saved values from a file, needed to transform test images

    pp_head('Creating and compiling model...')
    if config['LOAD_MODEL']:
        model = model_from_json(open('my_model_architecture.json').read())
    else:
        model = get_unet()
        json_string = model.to_json()
        open('my_model_architecture.json', 'w').write(json_string)
    
    if config['LOAD_LAST']:
        model.load_weights('unet.hdf5')

    if config['FIT']:
        pp_head('Fitting model...')
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
        model_checkpoint2 = ModelCheckpoint("weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', save_best_only=True)
        model.fit(imgs_train, imgs_mask_train,validation_split=config['VALIDATION'], batch_size=config['BATCH'], nb_epoch=config['EPOCH'], verbose=1, shuffle=True,
                  callbacks=[model_checkpoint,model_checkpoint2]) # batch size originally 32
    #else:
    #    model.test_on_batch(imgs_train, imgs_mask_train)

    pp_head(str(model.summary()))

    pp_head('Loading and preprocessing test data...')
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test, True)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= stats['mean']
    imgs_test /= stats['std']

    pp_head('Loading saved weights...')
    model.load_weights('unet.hdf5')

    pp_head('Predicting masks on test data...')
    imgs_mask_test = model.predict(imgs_test, verbose=1) # USe batch to speed up on large picture
    #imgs_mask_test = model.predict(imgs_test,1, verbose=1) # USe batch to speed up on large picture
    
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
