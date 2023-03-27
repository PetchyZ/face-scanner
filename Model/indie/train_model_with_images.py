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
model = load_model('facenet_keras.h5')
folder_train = 'D:/Dataset/train/'
folder_test = 'D:/Dataset/val/'
trainX, trainy = fp.load_dataset(folder_train)
testX, testy = fp.load_dataset(folder_test)

testX_faces = testX
newTrainX = list()
newTestX = list()
for face_pixels in testX:
    embedding = fp.get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
for face_pixels in trainX:
    embedding = fp.get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# prediction for the face
samples = np.expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])

# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()

filename = '/Face_recognition_Project/ModelExported_model/admin_model.sav'
pickle.dump(model, open(filename, 'wb'))
