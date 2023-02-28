import sys
sys.path.insert(0, '/Face_recognition_Project/Model/')
sys.path.insert(0, '/Face_recognition_Project/')
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from bson.binary import Binary
import face_processing as fp
import path_settings as ps
import numpy as np
import datetime
import pymongo
import base64
import os

# mongo db settings
try:
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["Face_recognition_project"]
	print("Database connection: successfully")
except:
	print("Can't connect to database")

def list_criminal():
	try:
		ls = list()
		for coll_name in mydb.list_collection_names():
			if coll_name != "user" and coll_name != "model":
				print("collection:{}".format(coll_name))
				ls.append(coll_name)
			else:
				print("This is a {} category".format(coll_name))
		return ls
	except:
		print("List Criminal Error")

def list_criminal_profile():
    try:
        ls_images = {}
        for coll_name in mydb.list_collection_names():
            if coll_name != "user" and coll_name != "model":
                mycol = mydb[coll_name]
                myquery = { "name": coll_name }
                mydoc = mycol.find_one(myquery)
                face_arr = np.array(mydoc['face_image'])
                #print("collection:{} \n face iamge: {}".format(coll_name,face_arr))
                #print("face array type:{}".format(type(face_arr)))
                ls_images[coll_name] = face_arr
                
            else:
                print("This is a {} category".format(coll_name))
        return ls_images
    except:
        print("List Criminal Profile Error")
        
def list_criminal_profile_b64():
    try:
        for coll_name in mydb.list_collection_names():
            if coll_name != "user" and coll_name != "model":
                mycol = mydb[coll_name]
                myquery = { "name": coll_name }
                mydoc = mycol.find_one(myquery)
                face_arr = np.array(mydoc['face_image'])
                
                #print("collection:{} \n face iamge: {}".format(coll_name,face_arr))
                #print("face array type:{}".format(type(face_arr)))
                b64_image = base64.b64encode(face_arr)
                #b64_image = face_arr.tostring()
                #base 64
            else:
                print("This is a {} category".format(coll_name))
        return b64_image.decode("utf-8")
    
    except:
        print("List Criminal Profile Error")    
def delete_criminal(name):
	cr_name = name
	mycol = mydb[cr_name]
	mycol.drop()
	print("Delete successfully")
 
def get_model(path):
    print("Getting model")
    mycol = mydb["model"]
    mydict = {"model name": "full_face_model"}
    data=mycol.find_one(mydict)
    with open(path + 'full_face_model.sav', "wb") as f:
        f.write((data['model_file']))
        
    mydict = {"model name": "half_face_model"}
    data=mycol.find_one(mydict)
    with open(path + 'half_face_model.sav', "wb") as f:
        f.write((data['model_file']))
    print("Get model completed")
    
def delete_model():
    for coll in mydb.list_collection_names():
        if coll == 'model':
            print(coll)
            mycol = mydb[coll]
            mycol.drop()
    print("Delete successfully")
 
def upload_model(path):
    print("Start Insert model")
    
    with open(path + '/full_face_model.sav', "rb") as f:
        encoded = Binary(f.read())
    
    #pickle.load(open(folder_dataset+filename, 'rb'))
    mycol = mydb["model"]
    #mycol = mydb["{}".format(collection_name = j)]
    mydict = {"model name": "full_face_model"
                , "model_file":encoded
                ,"threshold":1.250
                ,"created_time":datetime.datetime.now()}
    mycol.insert_one(mydict)
    
    with open(path + '/half_face_model.sav', "rb") as f:
        encoded = Binary(f.read())
    mydict = {"model name": "half_face_model"
                , "model_file":encoded
                ,"threshold":1.250
                ,"created_time":datetime.datetime.now()}
    mycol.insert_one(mydict)
    print("Insert model sucessfully")

def user_update():
    get_model(ps.get_exported_model_folder())
    
def send_user_data(username,password,role):
	try:
		mycol = mydb["user"]
		mydict = {"username": username.value, "password": password.value, "role": role.value}
		mycol.insert_one(mydict)
		print("Insert successfully")
	except:
		print("Send data error")
		
def upload_face_criminal(directory_path):
    folder_dataset = directory_path
    #folder_for_train = folder_dataset+"/criminal_profile/"
    trainX, trainy = fp.load_dataset(folder_dataset)
    for i, j in zip(trainX, trainy):
        mycol = mydb["{collection_name}".format(collection_name = j)]
        mydict = {"name": j, "face_image":i.tolist()}
        mycol.insert_one(mydict)
        #print("This is shape:",i.shape)
    print("Insert sucessfully")
    
def check_user_data(username):
	try:
		mycol = mydb["user"]
		mydict = {"username": username}
		if mycol.find_one(mydict):
			return True
		else:
			return False
	except:
		print("Check data error")