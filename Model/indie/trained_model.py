import pickle
import numpy as np
import face_processing as fp
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from os import listdir
from os.path import isdir
from numpy import asarray
from mtcnn import MTCNN
from numpy import load
from PIL import Image

def extract_cam_face(image, result, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = result
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def predict_face(image, result, model, loaded_model):
    face_pixels = extract_cam_face(image, result)
    # test model on a random example from the test dataset
    face_emb = get_embedding(model, face_pixels)
    samples = expand_dims(face_emb, axis=0)
    # prediction for the face
    yhat_class = loaded_model.predict(samples)
    yhat_prob = loaded_model.predict_proba(samples)
    return yhat_class