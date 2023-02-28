import numpy as np
import pymongo
import os

# mongo db settings
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Face_recognition_project"]
mycol = mydb["user"]

def check(username_check,password_check):
    try:
        print("username:",username_check.value)
        print("password:",password_check.value)
        print("role:",mycol.find_one({'username':username_check.value,'password':password_check.value})['role'])
        print("Object query:",mycol.find_one({'username':username_check.value,'password':password_check.value}))
        if mycol.find_one({'username':username_check.value,'password':password_check.value})['role'] == 'admin':
            return "admin"
        elif mycol.find_one({'username':username_check.value,'password':password_check.value})['role'] == 'user':
            return "user"
        else:
            return False
    except:
        print("Don't have any user")
    # for r in mydb[mycol]:
    #     print("Get user")
    #   ls.append(r['vector'])
    # if os.path.exists(path) == False:
    #     np.save(path,np.array(ls))
    # print("Directory '% s' created" % coll_name)
    # print("numpy shape:",np.array(ls).shape)
    # ls = list()
    # print('\n')