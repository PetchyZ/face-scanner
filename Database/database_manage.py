import datetime
#from cryptography.fernet import Fernet
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.decimal128 import Decimal128
from bson.binary import Binary
import Model.face_processing as fp
import path.path_settings as ps
import numpy as np
import datetime
import pymongo
import secrets
import string
import base64
import cv2
import os

#categories
non_criminal = ['contact_info','model','user']

# mongo db settings
try:
	# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	# mydb = myclient["Face_recognition_project"]
    myclient = pymongo.MongoClient("mongodb+srv://parit:132456@cluster1.gktrrcl.mongodb.net/?retryWrites=true&w=majority")
    mydb = myclient["Face_recognition_project"]
    print("Database connection: successfully")
except:
	print("Can't connect to database")

def get_super_admin():
    mycol = mydb["user"]
    mydict = {'username':'admin'}
    print(mycol.find_one(mydict))
    return mycol.find_one(mydict)
    
def get_ready():
    #clear data
    #frame cap
    dir = ps.get_frame_cap_path()
    if os.listdir(dir): 
        for f in os.listdir(dir):
            if f != 'frame_cap_description.txt':
                os.remove(os.path.join(dir, f))
    
    #criminal images
    dir = ps.get_assets_folder() + "criminal_images/"
    if os.listdir(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir,f))

    #get data
    download_face_criminals()
    model_update()
    print("Get ready complete")

         

def clear_frame_cap():
    dir = ps.get_frame_cap_path()
    if os.listdir(dir): 
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    else:
        print("frame_cap is empty.")

# def encrypting_password(password):
#     key = Fernet.generate_key()
#     fernet = Fernet(key)
#     enc_password = fernet.encrypt(password.encode())
#     print("Encrpting password:",enc_password)
#     return enc_password

# #Error
# def decrypting_password(password):
#     try:
#         key = Fernet.generate_key()
#         fernet = Fernet(key)
#         print("User password:",password)
#         dec_password = fernet.decrypt(password).decode()
#         print("Decrpting password:",dec_password)
#         return dec_password
#     except Exception as e:
#         print("decrypting error")
#         print(e)

def generate_password():
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for i in range(20)) 
    return password

def add_first_user():
    mycol = mydb["user"]
    if len(list(mycol.find())) != 0:
        print("User collection already created.")
        return False
    else:
        password = generate_password()
        mydict = {'username':'admin','password':password,'role':'super_admin'}
        mycol.insert_one(mydict)
        print("Username: ",'admin')   
        print("Password: ",password)  
        print("Admin have been added.")
        return True

def get_threshold():
    mycol = mydb["model"]
    mydict = {"model name": "full_face_model"}
    full_model_threshold=mycol.find_one(mydict)
    mydict = {"model name": "half_face_model"}
    half_model_threshold=mycol.find_one(mydict)
    print(full_model_threshold['threshold'],half_model_threshold['threshold'])
    return full_model_threshold['threshold'].to_decimal(),half_model_threshold['threshold'].to_decimal()

def check_non_criminal_categories(item):
    if item not in non_criminal:
        return True
    return False

def list_contact():
    ls=list()
    mycol = mydb["contact_info"]
    for i in mycol.find():
        ls.append({"id":i['_id'],"name":i['contact_first_name']+" "+i['contact_last_name']})
    return ls

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

def list_criminal_profile(profile_name):
    if profile_name not in non_criminal:
        criminal_info = mydb[profile_name].find()[0]
        contact_info = search_contact_by_id(criminal_info['contact_id'])
        criminal_info.pop('contact_id')
        contact_info.pop('_id')
        criminal_info.update(contact_info)
        print("list criminal profile succesful")
        return criminal_info
        
        
def list_criminal_profile_b64():
    try:
        for coll_name in mydb.list_collection_names():
            if coll_name != "user" and coll_name != "model":
                mycol = mydb[coll_name]
                myquery = { "name": coll_name }
                mydoc = mycol.find_one(myquery)
                face_arr = np.array(mydoc['face_image'])
                b64_image = base64.b64encode(face_arr)
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

def delete_contact(obj_id):
	mycol = mydb['contact_info']
	mycol.delete_one({'_id':ObjectId(obj_id)})
	print("Delete successfully")

def check_contact_with_criminal(obj_id):
    for coll_name in mydb.list_collection_names():
        if check_non_criminal_categories(coll_name):
            if mydb[coll_name].find_one({},{'contact_id':1})['contact_id'] == obj_id:
                #have relation
                return False
    #Non relation
    return True

def get_model(path):
    print("Getting model")
    if "model" in mydb.list_collection_names():
        mycol = mydb["model"]
        mydict = {"model name": "full_face_model"}
        data=mycol.find_one(mydict)
        with open(path + 'full_face_model.sav', "wb") as f:
            f.write((data['model_file']))
        with open(ps.get_full_face_classes_path(), "wb") as f:
            f.write((data['classes']))

        mydict = {"model name": "half_face_model"}
        data=mycol.find_one(mydict)
        with open(path + 'half_face_model.sav', "wb") as f:
            f.write((data['model_file']))
        with open(ps.get_half_face_classes_path(), "wb") as f:
            f.write((data['classes']))

        print("Get model completed")
    
def delete_model():
    for coll in mydb.list_collection_names():
        if coll == 'model':
            print(coll)
            mycol = mydb[coll]
            mycol.drop()
    print("Delete successfully")
 
def upload_model(path,fm_threshold,hm_threshold):
    print("Start Insert model")
    print("FF threshold:",fm_threshold)
    print("HF threshold:",hm_threshold)
    if 'model' in mydb.list_collection_names():
        mydb['model'].drop()

    with open(path + '/full_face_model.sav', "rb") as f:
        encoded = Binary(f.read())
    
    with open(path + '/classes_full_face.npy', "rb") as f:
        encoded_class = Binary(f.read())

    mycol = mydb["model"]
    mydict = {"model name": "full_face_model"
                , "model_file":encoded
                ,"threshold":Decimal128(fm_threshold)
                ,"created_time":datetime.datetime.now()
                ,"classes":encoded_class
            }
    mycol.insert_one(mydict)
    
    with open(path + '/half_face_model.sav', "rb") as f:
        encoded = Binary(f.read())

    with open(path + '/classes_half_face.npy', "rb") as f:
        encoded_class = Binary(f.read())

    mydict = {"model name": "half_face_model"
                , "model_file":encoded
                ,"threshold":Decimal128(hm_threshold)
                ,"created_time":datetime.datetime.now()
                ,"classes":encoded_class
            }
    mycol.insert_one(mydict)
    print("Insert model sucessfully")

def model_update():
    get_model(ps.get_exported_model_folder())
    
def send_user_data(username,password,role):
    try:
        mycol = mydb["user"]
        mydict = {"username": username.value, "password": password.value, "role": role.value}
        mycol.insert_one(mydict)
        print("Insert successfully")
    except Exception as e:
        print("Send user data Error:")
        print(e)

def upload_face_criminal(directory_path):
    folder_dataset = directory_path
    trainX, trainy = fp.load_dataset(folder_dataset)
    for i, j in zip(trainX, trainy):
        mycol = mydb["{collection_name}".format(collection_name = j)]
        mydict = {"name": j, "face_image":i.tolist()}
        mycol.insert_one(mydict)
    print("Insert sucessfully")

def download_face_criminals():
    assets_folder = ps.get_assets_folder()
    print("list :",mydb.list_collection_names())
    if mydb.list_collection_names():
        for coll_name in mydb.list_collection_names():
            if check_non_criminal_categories(coll_name):
                mycol = mydb[coll_name]
                img = np.array(mycol.find_one({},{"_id":0, "face_image":1})["face_image"])
                cv2.imwrite(assets_folder+"criminal_images/"+"{}.png".format(coll_name),img)
                img = cv2.imread(assets_folder+"criminal_images/"+"{}.png".format(coll_name))
                cv2.imwrite(assets_folder+"criminal_images/"+"{}.png".format(coll_name),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("Download sucessfully")
    else:
         print("Collections is empty")
    
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
                
def add_criminal_data(cri_fn,cri_ln,cri_CID,cri_nation,cri_age,cri_sex,cri_offense,selected_file_image,contact_id,cri_announced_date):
    try:
        date = cri_announced_date.split('/')
        print("contact id:",contact_id)
        trainX = fp.extract_face(selected_file_image)
        mycol = mydb["{criminal_info}".format(criminal_info = cri_fn+"_"+cri_ln)]
        mydict = {"criminal_first_name":cri_fn,
                  "criminal_last_name":cri_ln,
                  "criminal_citizen_id":cri_CID,
                  "criminal_nationality":cri_nation,
                  "criminal_age":Int64(cri_age),
                  "criminal_sex":cri_sex,
                  "criminal_offense":cri_offense,
                  "face_image":trainX.tolist(),
                  "contact_id":contact_id,
                  "criminal_anounced_date":datetime.datetime(int(date[2]),int(date[1]),int(date[0]))
                  }
        mycol.insert_one(mydict)
        print("Insert image sucessfully")
        print("Update criminal info...")
        download_face_criminals()
        print("Update criminal sucessfully")
        
    except:
        print("Error: can't input criminal data")

def add_contact_data(contact_fn,contact_ln,contact_email,contact_num,contact_lo,contact_pas_code,contact_add_no,contact_vil_no,contact_road,contact_sub_dist,contact_dist,contact_provi):
    try:
        mycol = mydb['contact_info']
        mydict = {"contact_first_name":contact_fn,
                  "contact_last_name":contact_ln,
                  "contact_email":contact_email,
                  "contact_number":contact_num,
                  "contact_location":contact_lo,
                  "contact_pastal_code":contact_pas_code,
                  "contact_address_no":contact_add_no,
                  "contact_village_no":contact_vil_no,
                  "contact_road":contact_road,
                  "contact_sub_district":contact_sub_dist,
                  "contact_district":contact_dist,
                  "contact_province":contact_provi
        }
        mycol.insert_one(mydict)
    except:
        print("Error: can't input contact data")

def search_contact_by_id(contact_id):
    mycol = mydb['contact_info']
    contact_info = mycol.find_one({'_id':ObjectId(contact_id)})
    return contact_info

