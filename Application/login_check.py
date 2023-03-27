import pymongo

# mongo db settings
myclient = pymongo.MongoClient("mongodb+srv://parit:132456@cluster1.gktrrcl.mongodb.net/?retryWrites=true&w=majority")
mydb = myclient["Face_recognition_project"]
mycol = mydb["user"]

def check(username_check,password_check):
    try:
        print("username:",username_check.value)
        print("password:",password_check.value)
        print("role:",mycol.find_one({'username':username_check.value})['role'])
        print("Object query:",mycol.find_one({'username':username_check.value}))
        if mycol.find_one({'username':username_check.value,'password':password_check.value})['role'] == 'super_admin':
            return "super_admin"
        elif mycol.find_one({'username':username_check.value,'password':password_check.value})['role'] == 'admin':
            return "admin"
        elif mycol.find_one({'username':username_check.value,'password':password_check.value})['role'] == 'user':
            return "user"
        else:
            return False
    except Exception as e:
        print("Login check Error:")
        print(e)
 