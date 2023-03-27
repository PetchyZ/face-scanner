import os
import pickle
import numpy as np
import face_processing as fp
from PIL import Image
from mtcnn import MTCNN
from random import choice
from os import listdir
from os.path import isdir
from matplotlib import pyplot
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

#train,test path (important)
#model = load_model('facenet_keras.h5')
folder_train_np = '/Face_recognition_Project/Database/train_data/'
print("list directory:",os.listdir(folder_train_np))
for name in os.listdir(folder_train_np):
    trainX = np.load(folder_train_np+name)
    print("name:",os.path.splitext(name)[0])
    print(trainX,"\n\n")

trainX_new = list()
trainy = list()
for name in os.listdir(folder_train_np):
    for elm in trainX:
        trainX_new.append(elm)
        trainy.append(name)
trainX = np.array(trainX_new)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

filename = '/Face_recognition_Project/ModelExported_model/user_model.sav'
pickle.dump(model, open(filename, 'wb'))
