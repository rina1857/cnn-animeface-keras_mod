import numpy as np
np.random.seed(20160715) # シード値を固定
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Dropout, Flatten ##add
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.optimizers import SGD
#import sklearn.cross_validation
from sklearn.model_selection import train_test_split ##update

X_test=np.load('anime_face_data.npy')
Y_target=np.load('anime_face_target.npy')

#a_train, a_test, b_train, b_test = sklearn.cross_validation.train_test_split(X_test,Y_target) ##update
a_train, a_test, b_train, b_test = train_test_split(X_test,Y_target)

model = Sequential()

model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(203))
model.add(Activation('softmax'))

init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
lrs = LearningRateScheduler(0.01)

hist = model.fit(a_train,b_train, 
                batch_size=128, 
                nb_epoch=50, 
                validation_split=0.1, 
                verbose=1)

model_json_str = model.to_json()
open('anime_face_model.json', 'w').write(model_json_str)
model.save_weights('anime_face_model.h5')

score=model.evaluate(a_test, b_test, verbose=0)
print(score[1])
