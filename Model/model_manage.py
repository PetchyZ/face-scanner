import joblib
import os
import pickle
import numpy as np
import pandas as pd
import path.path_settings as ps
import Model.face_processing as fp
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image
from mtcnn import MTCNN
from random import choice
from itertools import product
from matplotlib import pyplot
import matplotlib.image as mpimg
from scipy.spatial import distance
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn import tree
import cv2

def close_event():
    pyplot.close()

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def make_positive(idendities):
    positives = []
    for key, values in idendities.items():
        for i in range(0, len(values)-1):
            for j in range(i+1, len(values)):
                positive = []
                positive.append(values[i])
                positive.append(values[j])
                positives.append(positive)
        
    positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
    positives["decision"] = "Yes"
    return positives

def make_negative(idendities):
    samples_list = list(idendities.values())
    negatives = []
    for i in range(0, len(idendities) - 1):
        for j in range(i+1, len(idendities)):
            cross_product = product(samples_list[i], samples_list[j])
            cross_product = list(cross_product)
            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                negatives.append(negative)
    negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
    negatives["decision"] = "No"
    return negatives

def l2_normalize(x):
	return x / np.sqrt(np.sum(np.multiply(x, x)))

def extract_face(image, result, required_size=(160, 160)):
	pixels = np.asarray(image)
	x1, y1, width, height = result
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array

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

def train_full_face_model(path,parameters):
	print(parameters)
	#train,test path (important)
	model = load_model(ps.get_facenet_model_path())
	folder_train = path + "/train/"
	folder_test = path + "/val/"
	print("folder_train:",folder_train)
	print("folder_test:",folder_test)
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

	# test model on a random example from the test dataset
	selection = choice([i for i in range(testX.shape[0])])
	random_face_pixels = testX_faces[selection]
	random_face_emb = testX[selection]
	random_face_class = testy[selection]
	random_face_name = out_encoder.inverse_transform([random_face_class])

	# prediction for the face
	samples = np.expand_dims(random_face_emb, axis=0)

	#fit by grid
	param_grid = {}
	param_grid['C'] = parameters['c_values']
	param_grid['gamma'] = parameters['gamma_values']
	param_grid['kernel'] = parameters['kernel']
	grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=2)
	grid.fit(trainX,trainy)
	grid_predictions = grid.predict(testX)
	print("Best eatimator:",grid.best_estimator_)
	print("Confusion matrix:",confusion_matrix(testy,grid_predictions))
	print("Report:\n",classification_report(testy,grid_predictions))
	
	yhat_class = grid.predict(samples)
	yhat_prob = grid.predict_proba(samples)

	# get name
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	print('Expected: %s' % random_face_name[0])

	# plot for fun
	fig = pyplot.figure()
	timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
	timer.add_callback(close_event)
	pyplot.imshow(random_face_pixels)
	title = 'Full face Random person: %s \nAccuracy: (%.3f)' % (predict_names[0], class_probability)
	pyplot.title(title)
	timer.start()
	pyplot.show()

	filename = 'full_face_model.sav'
	pickle.dump(grid, open(ps.get_exported_model_folder() + filename, 'wb'))

	np.save(ps.get_exported_model_folder() + 'classes_full_face.npy', out_encoder.classes_)
	print("Model was Exported.")
	print("Train sucessfully")

	#find threshold
	c4_threshold, sigma2_threshold, sigma3_threshold = find_full_face_threshold(path+"/train",newTrainX)
	print(c4_threshold)

	return {'confusion_matrix': confusion_matrix(testy,grid_predictions),
			'classification_report': classification_report(testy,grid_predictions),
			'c4_threshold' : c4_threshold,
			'sigma2_threshold': sigma2_threshold,
			'sigma3_threshold':sigma3_threshold
			}
	
def train_half_face_model(path,parameters):
	model = load_model(ps.get_facenet_model_path())
	trainX, trainy = fp.load_dataset_half(path + "/train/")
	testX, testy = fp.load_dataset_half(path + "/val/")

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

	# test model on a random example from the test dataset
	selection = choice([i for i in range(testX.shape[0])])
	random_face_pixels = testX_faces[selection]
	random_face_emb = testX[selection]
	random_face_class = testy[selection]
	random_face_name = out_encoder.inverse_transform([random_face_class])

	# prediction for the face
	samples = np.expand_dims(random_face_emb, axis=0)

	#test grid
	param_grid = {}
	param_grid['C'] = parameters['c_values']
	param_grid['gamma'] = parameters['gamma_values']
	param_grid['kernel'] = parameters['kernel']
	grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=2)
	grid.fit(trainX,trainy)
	grid_predictions = grid.predict(testX)
	print("Best eatimator:",grid.best_estimator_)
	print("Confusion matrix:",confusion_matrix(testy,grid_predictions))
	print("Report:")
	print(classification_report(testy,grid_predictions))
	
	yhat_class = grid.predict(samples)
	yhat_prob = grid.predict_proba(samples)

	# get name
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	print('Expected: %s' % random_face_name[0])

	# plot for fun
	fig = pyplot.figure()
	timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
	timer.add_callback(close_event)
	pyplot.imshow(random_face_pixels)
	title = 'Half face %s (%.3f)' % (predict_names[0], class_probability)
	pyplot.title(title)
	timer.start()
	pyplot.show()

	filename = 'half_face_model.sav'
	joblib.dump(grid, ps.get_exported_model_folder() + filename)
	np.save(ps.get_exported_model_folder() + 'classes_half_face.npy', out_encoder.classes_)

	#find threshold
	c4_threshold, sigma2_threshold, sigma3_threshold = find_half_face_threshold(path+"/train",newTrainX)
	print(c4_threshold)

	return {'confusion_matrix': confusion_matrix(testy,grid_predictions),
			'classification_report': classification_report(testy,grid_predictions),
			'c4_threshold' : c4_threshold,
			'sigma2_threshold': sigma2_threshold,
			'sigma3_threshold':sigma3_threshold
			}

def predict_face(image, result, model, loaded_model):
	face_pixels = extract_cam_face(image, result)
	# test model on a random example from the test dataset
	face_emb = fp.get_embedding(model, face_pixels)
	samples = np.expand_dims(face_emb, axis=0)
	in_encoder = Normalizer(norm='l2')
	samples = in_encoder.transform(samples)
	# prediction for the face
	yhat_class = loaded_model.predict(samples)
	yhat_prob = loaded_model.predict_proba(samples)
	return yhat_class, samples, yhat_prob


def predict_face_mask_face(array_image, model, loaded_model):
	#face_pixels = extract_cam_face(image, result)
	face_pixels = array_image
	# test model on a random example from the test dataset
	face_emb = fp.get_embedding(model, face_pixels)
	samples = np.expand_dims(face_emb, axis=0)
	in_encoder = Normalizer(norm='l2')
	samples = in_encoder.transform(samples)
	# prediction for the face
	yhat_class = loaded_model.predict(samples)
	yhat_prob = loaded_model.predict_proba(samples)
	return yhat_class, yhat_prob

def test_full_image(image_path,model_path):
	model = load_model(ps.get_facenet_model_path())
	image_show=image_path
	
	#model path
	filename = model_path
	loaded_model = pickle.load(open(filename, 'rb'))

	#load classes
	out_encoder = LabelEncoder()
	out_encoder.classes_ = np.load(ps.get_full_face_classes_path())

	detector = MTCNN()
	image = cv2.imread(image_show)
	location = detector.detect_faces(image)
	if len(location) > 0:
		for face in location:
			face_pixels = extract_face(image,face['box'])
	else:
		print("Can't detect faces")
	# test model on a random example from the test dataset
	face_emb = fp.get_embedding(model, face_pixels)
	samples = np.expand_dims(face_emb, axis=0)
	# prediction for the face
	yhat_class = loaded_model.predict(samples)
	yhat_prob = loaded_model.predict_proba(samples)
	class_probability = yhat_prob[0,yhat_class[0]] * 100
	image_show = mpimg.imread(image_show)
	predict_names = out_encoder.inverse_transform(yhat_class)
	pyplot.imshow(image_show)
	title = 'Person: {per} Accuracy: ({prob: .3f})'.format(per=predict_names[0], prob=class_probability)
	pyplot.title(title)
	pyplot.show()
	
	return yhat_class

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def test_half_image(image_path,model_path):
	model = load_model(ps.get_facenet_model_path())
	image_show=image_path
	
	#model path
	filename = model_path
	loaded_model = joblib.load(open(filename, 'rb'))

	#load classes
	out_encoder = LabelEncoder()
	out_encoder.classes_ = np.load(ps.get_full_face_classes_path())

	detector = MTCNN()
	image = cv2.imread(image_show)
	location = detector.detect_faces(image)
	if len(location) > 0:
		for face in location:
			face_pixels = extract_face(image,face['box'])
	else:
		print("Can't detect faces")
	# test model on a random example from the test dataset
	face_emb = fp.get_embedding(model, face_pixels)
	samples = np.expand_dims(face_emb, axis=0)
	# prediction for the face
	yhat_class = loaded_model.predict(samples)
	yhat_prob = loaded_model.predict_proba(samples)
	class_probability = yhat_prob[0,yhat_class[0]] * 100
	image_show = mpimg.imread(image_show)
	predict_names = out_encoder.inverse_transform(yhat_class)
	pyplot.imshow(image_show)
	title = 'Person: {per} Accuracy: ({prob: .3f})'.format(per=predict_names[0], prob=class_probability)
	pyplot.title(title)
	pyplot.show()
	
	return yhat_class

def find_full_face_threshold(dataset_path,faces_emd):
	path = dataset_path
	count = 0
	list_of_label = []
	label = []
	face_arr = []
	#model = DeepFace.build_model("Facenet512")
	all_name_file = []
	for root, dir, file in os.walk(path):
		if count == 0 :
			list_of_label = dir
		for i in range(len(list_of_label)) :
			if root.find(list_of_label[i]) != -1:
				for j in file:
					label.append(list_of_label[i])
					#face_arr.append(fp.extract_face(root+'\\'+j))
					all_name_file.append(root+'\\'+j)
					print(root+'\\'+j)
		count += 1
	all_name_file = np.asarray(all_name_file)
	facenet_model = tf.keras.models.load_model(ps.get_facenet_model_path())
	emd = []

	emd = np.asarray(emd)
	i=0
	emd = faces_emd
	name_key = []
	val = []
	tmp_val = []
	count = 0
	for j in range(len(label)):
		if label[j] in name_key :
			tmp_val.append(emd[j])
		elif label[j] not in name_key :
			name_key.append(label[j])
			if count == 0 :
				print("IIIIIIIIIIIIIIIIIIIIIIIII")
				tmp_val.append(emd[j])
			else :
				print(len(tmp_val))
				val.append(tmp_val)
				tmp_val = []
				tmp_val.append(emd[j])
			print(len(tmp_val))
			count += 1
		if j == len(label) - 1 :
			val.append(tmp_val)
	val = np.array(val)
	name_key = np.array(name_key)
	print(len(name_key))
	print(len(val))
	print(val.shape)

	iden = {}

	for index,value in zip(name_key,val):
		iden[index] = value

	print(iden)
	print(len(val))
	print(val.shape)
	positives = make_positive(iden)
	negatives = make_negative(iden)
	df = pd.concat([positives, negatives]).reset_index(drop = True)
	instances = df[["file_x", "file_y"]].values.tolist()
	distances = []
	print("Type")
	print(type(str(instances[i][0])))
	print(str(instances[i][0]))
	for i in range(len(instances)):
		t_1 = l2_normalize(instances[i][0])
		t_2 = l2_normalize(instances[i][1])
		distances.append(findCosineSimilarity(t_1,t_2))
	df["distance"] = distances
	dt = []
	d_f = df.copy()
	print("####################   instance   ####################")
	print(l2_normalize(instances[0][0]))
	print(l2_normalize(instances[0][0]).shape)
	print("########################################")
	for i in range(len(instances)):
		t_1 = l2_normalize(instances[i][0])
		t_2 = l2_normalize(instances[i][1])
		dt.append(distance.euclidean(t_1,t_2))
	d_f["distance"] = dt
	print(d_f.head())
	tp_mean = round(df[df.decision == "Yes"].mean().values[0], 4)
	tp_std = round(df[df.decision == "Yes"].std().values[0], 4)
	fp_mean = round(df[df.decision == "No"].mean().values[0], 4)
	fp_std = round(df[df.decision == "No"].std().values[0], 4)
	tp_mean_2 = round(d_f[d_f.decision == "Yes"].mean().values[0], 4)
	tp_std_2 = round(d_f[d_f.decision == "Yes"].std().values[0], 4)

	print("Mean of true positives " + str(tp_mean))
	print("Std of true positives " + str(tp_std))
	print("Mean of false positives " + str(fp_mean))
	print("Std of false positives " + str(fp_std))

	data_tmp = d_f.copy()
	decision_table = data_tmp[['distance', 'decision']].rename(columns = {"decision": "decision"}).copy()
	#decision_table.to_csv(r'Data_threshold_full_face.csv', index = False)

	a_ = d_f[d_f.decision == "Yes"].distance
	b_ = d_f[d_f.decision == "No"].distance
	Euclidean_a_max = max(a_)
	Euclidean_a_min = min(b_)
	print("max of Positive is",Euclidean_a_max)
	print("min of Negative is",Euclidean_a_min)
	threshold_st = round(tp_mean + 2 * tp_std, 4)
	threshold_st_2 = tp_mean_2 + 2 * tp_std_2
	threshold_st_3 = tp_mean_2 + 3 * tp_std_2
	print("Static Approach sigma 2 cosine similarity")
	print(threshold_st)
	print("##############################")
	print("Static Approach sigma 2 -euclidean")
	print(threshold_st_2)
	print("##############################")
	print("Static Approach sigma 3 -euclidean")
	print(threshold_st_3)
	print("##############################")
	# Plot
	ax1 = d_f[d_f.decision == "Yes"].distance.plot.kde()
	ax2 = d_f[d_f.decision == "No"].distance.plot.kde()
	sns.kdeplot(d_f[d_f.decision == "Yes"].distance)
	sns.kdeplot(d_f[d_f.decision == "No"].distance)
	#ax2.axvline(result, color='red')
	plt.show()
	
	clf = tree.DecisionTreeClassifier(max_depth=1)
	clf = clf.fit(decision_table['distance'].values.reshape(-1,1), decision_table['decision'])
	# print("Predict: ",clf.predict([0.843420]))
	print(tree.export_text(clf))
	tree_text = tree.export_text(clf)
	threshold_c4 = float(tree.export_text(clf)[tree_text.find("<= ")+3:tree_text.find("|", tree_text.find("<= "))])
	threshold_2sigma = threshold_st_2
	threshold_3sigma = threshold_st_3

	print("C4.5 Threshold:",threshold_c4)
	print("Sigma2 Threshold:",threshold_2sigma)
	print("Sigma3 Threshold:",threshold_3sigma)

	decision_table_c4 = decision_table.copy()
	decision_table_c4["prediction"] = "No" #init
	idx = decision_table_c4[decision_table_c4.distance <= threshold_c4].index
	decision_table_c4.loc[idx, 'prediction'] = 'Yes'
	
	decision_table_2sigma = decision_table.copy()
	decision_table_2sigma["prediction"] = "No" #init
	idx = decision_table_2sigma[decision_table_2sigma.distance <= threshold_2sigma].index
	decision_table_2sigma.loc[idx, 'prediction'] = 'Yes'

	decision_table_3sigma = decision_table.copy()
	decision_table_3sigma["prediction"] = "No" #init
	idx = decision_table_3sigma[decision_table_3sigma.distance <= threshold_3sigma].index
	decision_table_3sigma.loc[idx, 'prediction'] = 'Yes'

	cm_c4 = confusion_matrix(decision_table_c4['decision'].values, decision_table_c4['prediction'].values)
	cm_2sigma = confusion_matrix(decision_table_2sigma['decision'].values, decision_table_2sigma['prediction'].values)
	cm_3sigma = confusion_matrix(decision_table_3sigma['decision'].values, decision_table_3sigma['prediction'].values)
    
	tn, false_p, fn, tp = cm_c4.ravel()
	recall = tp / (tp + fn)
	precision = tp / (tp + false_p)
	accuracy = (tp + tn)/(tn + false_p +  fn + tp)
	f1 = 2 * (precision * recall) / (precision + recall)

	tn2, fp2, fn2, tp2 = cm_2sigma.ravel()
	recall2 = tp2 / (tp2 + fn2)
	precision2 = tp2 / (tp2 + fp2)
	accuracy2 = (tp2 + tn2)/(tn2 + fp2 +  fn2 + tp2)
	f1_2 = 2 * (precision2 * recall2) / (precision2 + recall2)

	tn3, fp3, fn3, tp3 = cm_3sigma.ravel()
	recall3 = tp3 / (tp3 + fn3)
	precision3 = tp3 / (tp3 + fp3)
	accuracy3 = (tp3 + tn3)/(tn3 + fp3 +  fn3 + tp3)
	f1_3 = 2 * (precision3 * recall3) / (precision3 + recall3)

	print("Threshold C4.5")
	print(cm_c4)
	print("Precision: ", 100*precision,"%")
	print("Recall: ", 100*recall,"%")
	print("F1 score ",100*f1, "%")
	print("Accuracy: ", 100*accuracy,"%")

	print("Threshold 2 sigma")
	print(cm_2sigma)
	print("Precision: ", 100*precision2,"%")
	print("Recall: ", 100*recall2,"%")
	print("F1 score ",100*f1_2, "%")
	print("Accuracy: ", 100*accuracy2,"%")

	print("Threshold 3 sigma")
	print(cm_3sigma)
	print("Precision: ", 100*precision3,"%")
	print("Recall: ", 100*recall3,"%")
	print("F1 score ",100*f1_3, "%")
	print("Accuracy: ", 100*accuracy3,"%")
	return {"Threshold":threshold_c4, "Precision":100*precision, "Recall":100*recall, "F1":100*f1, "Accuracy":100*accuracy} , {"Threshold":threshold_2sigma, "Precision":100*precision2, "Recall":100*recall2, "F1":100*f1_2, "Accuracy":100*accuracy2}, {"Threshold":threshold_3sigma, "Precision":100*precision3,"Recall":100*recall3,"F1":f1_3*100,"Accuracy":accuracy3*100}

def find_half_face_threshold(dataset_path,faces_emd):
	path = dataset_path
	count = 0
	list_of_label = []
	label = []
	face_arr = []
	#model = DeepFace.build_model("Facenet512")
	all_name_file = []
	for root, dir, file in os.walk(path):
		if count == 0 :
			list_of_label = dir
		for i in range(len(list_of_label)) :
			if root.find(list_of_label[i]) != -1:
				for j in file:
					label.append(list_of_label[i])
					#face_arr.append(fp.extract_face(root+'\\'+j))
					all_name_file.append(root+'\\'+j)
					print(root+'\\'+j)
		count += 1
	all_name_file = np.asarray(all_name_file)

	i=0
	emd = faces_emd
	name_key = []
	val = []
	tmp_val = []
	count = 0
	for j in range(len(label)):
		if label[j] in name_key :
			tmp_val.append(emd[j])
		elif label[j] not in name_key :
			name_key.append(label[j])
			if count == 0 :
				print("IIIIIIIIIIIIIIIIIIIIIIIII")
				tmp_val.append(emd[j])
			else :
				print(len(tmp_val))
				val.append(tmp_val)
				tmp_val = []
				tmp_val.append(emd[j])
			print(len(tmp_val))
			count += 1
		if j == len(label) - 1 :
			val.append(tmp_val)
	val = np.array(val)
	name_key = np.array(name_key)
	print(len(name_key))
	print(len(val))
	print(val.shape)

	iden = {}

	for index,value in zip(name_key,val):
		iden[index] = value

	print(iden)
	print(len(val))
	print(val.shape)
	positives = make_positive(iden)
	negatives = make_negative(iden)
	df = pd.concat([positives, negatives]).reset_index(drop = True)
	instances = df[["file_x", "file_y"]].values.tolist()
	distances = []
	print("Type")
	print(type(str(instances[i][0])))
	print(str(instances[i][0]))
	for i in range(len(instances)):
		t_1 = l2_normalize(instances[i][0])
		t_2 = l2_normalize(instances[i][1])
		distances.append(findCosineSimilarity(t_1,t_2))
	df["distance"] = distances
	dt = []
	d_f = df.copy()
	print("####################   instance   ####################")
	print(l2_normalize(instances[0][0]))
	print(l2_normalize(instances[0][0]).shape)
	print("########################################")
	for i in range(len(instances)):
		t_1 = l2_normalize(instances[i][0])
		t_2 = l2_normalize(instances[i][1])
		dt.append(distance.euclidean(t_1,t_2))
	d_f["distance"] = dt
	print(d_f.head())
	tp_mean = round(df[df.decision == "Yes"].mean().values[0], 4)
	tp_std = round(df[df.decision == "Yes"].std().values[0], 4)
	fp_mean = round(df[df.decision == "No"].mean().values[0], 4)
	fp_std = round(df[df.decision == "No"].std().values[0], 4)
	tp_mean_2 = round(d_f[d_f.decision == "Yes"].mean().values[0], 4)
	tp_std_2 = round(d_f[d_f.decision == "Yes"].std().values[0], 4)

	print("Mean of true positives " + str(tp_mean))
	print("Std of true positives " + str(tp_std))
	print("Mean of false positives " + str(fp_mean))
	print("Std of false positives " + str(fp_std))

	data_tmp = d_f.copy()
	decision_table = data_tmp[['distance', 'decision']].rename(columns = {"decision": "decision"}).copy()
	#decision_table.to_csv(r'Data_threshold_full_face.csv', index = False)

	a_ = d_f[d_f.decision == "Yes"].distance
	b_ = d_f[d_f.decision == "No"].distance
	Euclidean_a_max = max(a_)
	Euclidean_a_min = min(b_)
	print("max of Positive is",Euclidean_a_max)
	print("min of Negative is",Euclidean_a_min)
	threshold_st = round(tp_mean + 2 * tp_std, 4)
	threshold_st_2 = tp_mean_2 + 2 * tp_std_2
	threshold_st_3 = tp_mean_2 + 3 * tp_std_2
	print("Static Approach sigma 2 cosine similarity")
	print(threshold_st)
	print("##############################")
	print("Static Approach sigma 2 -euclidean")
	print(threshold_st_2)
	print("##############################")
	print("Static Approach sigma 3 -euclidean")
	print(threshold_st_3)
	print("##############################")
	# Plot
	ax1 = d_f[d_f.decision == "Yes"].distance.plot.kde()
	ax2 = d_f[d_f.decision == "No"].distance.plot.kde()
	sns.kdeplot(d_f[d_f.decision == "Yes"].distance)
	sns.kdeplot(d_f[d_f.decision == "No"].distance)
	
	#ax2.axvline(result, color='red')
	plt.show()
	
	clf = tree.DecisionTreeClassifier(max_depth=1)
	clf = clf.fit(decision_table['distance'].values.reshape(-1,1), decision_table['decision'])
	# print("Predict: ",clf.predict([0.843420]))
	print(tree.export_text(clf))
	tree_text = tree.export_text(clf)
	threshold_c4 = float(tree.export_text(clf)[tree_text.find("<= ")+3:tree_text.find("|", tree_text.find("<= "))])
	threshold_2sigma = threshold_st_2
	threshold_3sigma = threshold_st_3

	print("C4.5 Threshold:",threshold_c4)
	print("Sigma2 Threshold:",threshold_2sigma)
	print("Sigma3 Threshold:",threshold_3sigma)

	decision_table_c4 = decision_table.copy()
	decision_table_c4["prediction"] = "No" #init
	idx = decision_table_c4[decision_table_c4.distance <= threshold_c4].index
	decision_table_c4.loc[idx, 'prediction'] = 'Yes'
	
	decision_table_2sigma = decision_table.copy()
	decision_table_2sigma["prediction"] = "No" #init
	idx = decision_table_2sigma[decision_table_2sigma.distance <= threshold_2sigma].index
	decision_table_2sigma.loc[idx, 'prediction'] = 'Yes'

	decision_table_3sigma = decision_table.copy()
	decision_table_3sigma["prediction"] = "No" #init
	idx = decision_table_3sigma[decision_table_3sigma.distance <= threshold_3sigma].index
	decision_table_3sigma.loc[idx, 'prediction'] = 'Yes'

	cm_c4 = confusion_matrix(decision_table_c4['decision'].values, decision_table_c4['prediction'].values)
	cm_2sigma = confusion_matrix(decision_table_2sigma['decision'].values, decision_table_2sigma['prediction'].values)
	cm_3sigma = confusion_matrix(decision_table_3sigma['decision'].values, decision_table_3sigma['prediction'].values)
    
	tn, false_p, fn, tp = cm_c4.ravel()
	recall = tp / (tp + fn)
	precision = tp / (tp + false_p)
	accuracy = (tp + tn)/(tn + false_p +  fn + tp)
	f1 = 2 * (precision * recall) / (precision + recall)

	tn2, fp2, fn2, tp2 = cm_2sigma.ravel()
	recall2 = tp2 / (tp2 + fn2)
	precision2 = tp2 / (tp2 + fp2)
	accuracy2 = (tp2 + tn2)/(tn2 + fp2 +  fn2 + tp2)
	f1_2 = 2 * (precision2 * recall2) / (precision2 + recall2)

	tn3, fp3, fn3, tp3 = cm_3sigma.ravel()
	recall3 = tp3 / (tp3 + fn3)
	precision3 = tp3 / (tp3 + fp3)
	accuracy3 = (tp3 + tn3)/(tn3 + fp3 +  fn3 + tp3)
	f1_3 = 2 * (precision3 * recall3) / (precision3 + recall3)

	print("Threshold C4.5")
	print(cm_c4)
	print("Precision: ", 100*precision,"%")
	print("Recall: ", 100*recall,"%")
	print("F1 score ",100*f1, "%")
	print("Accuracy: ", 100*accuracy,"%")

	print("Threshold 2 sigma")
	print(cm_2sigma)
	print("Precision: ", 100*precision2,"%")
	print("Recall: ", 100*recall2,"%")
	print("F1 score ",100*f1_2, "%")
	print("Accuracy: ", 100*accuracy2,"%")

	print("Threshold 3 sigma")
	print(cm_3sigma)
	print("Precision: ", 100*precision3,"%")
	print("Recall: ", 100*recall3,"%")
	print("F1 score ",100*f1_3, "%")
	print("Accuracy: ", 100*accuracy3,"%")
	return {"Threshold":threshold_c4, "Precision":100*precision, "Recall":100*recall, "F1":100*f1, "Accuracy":100*accuracy} , {"Threshold":threshold_2sigma, "Precision":100*precision2, "Recall":100*recall2, "F1":100*f1_2, "Accuracy":100*accuracy2}, {"Threshold":threshold_3sigma, "Precision":100*precision3,"Recall":100*recall3,"F1":f1_3*100,"Accuracy":accuracy3*100}
