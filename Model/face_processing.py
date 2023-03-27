import numpy as np
from PIL import Image
from mtcnn import MTCNN
from os import listdir
from os.path import isdir
from skimage.transform import resize

# extract a single face from a given photograph


def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = np.asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array

# load images and extract faces for all images in a directory


def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		path = directory + filename
		face = extract_face(path)
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + '/' +subdir + '/'
		print(path)
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)

# get the face embedding for one face


def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = np.expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def cut_mask_for_face_regconition(image):
	point_eye = 80
	X1 = image.copy()
	X2 = image.copy()
	X2[point_eye:160,:] = 0
	X3 = X1[0:point_eye,:]
	cut_face = resize(X3, (160, 160))
	#cut_face.reshape((160,160,3))
	return cut_face

def extract_face_half(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = np.asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	face_array = cut_mask_for_face_regconition(face_array)
	return face_array

def load_faces_half(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		path = directory + filename
		face = extract_face_half(path)
		faces.append(face)
	return faces

def load_dataset_half(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces_half(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)   
