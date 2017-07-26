import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
#import sklearn.cross_validation ##update
import cv2 as cv
np.random.seed(20160717)
from sklearn.model_selection import train_test_split

X_test=np.load('anime_face_data.npy')
Y_target=np.load('anime_face_target.npy')

model = model_from_json(open('anime_face_model.json').read())
model.load_weights('anime_face_model.h5')
init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])

image = cv.imread('ritsu.png')
image = cv.resize(image, (32, 32))
image = image.transpose(2,0,1)
image = image/255.

image=image.reshape(1,3,32,32)

for i in range(202):
    sample_target=np.array([i])
    score = model.evaluate(image, sample_target, verbose=0)
    if score[1]==1.0:
        break
print(i)