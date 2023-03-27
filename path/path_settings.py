#asset
assets_folder = r"./Application/assets/"

#Major path
project_path = r'.'

#Normal Model
exported_model_folder = r"./Model/Exported_model/"
facenet_model_path = r"./Model/FaceNet_model/facenet_keras.h5"
full_model_path = r"./Model/Exported_model/full_face_model.sav"
half_model_path = r"./Model/Exported_model/half_face_model.sav"
full_face_classes_path = r"./Model/Exported_model/classes_full_face.npy"
half_face_classes_path = r"./Model/Exported_model/classes_half_face.npy"
frame_cap_path = r"./Application/assets/frame_cap"

#Mask Model
mask_detect_model_folder = r'./Model/Mask_model/learnTensorFlow-main/faceMaskDetection'
p_model = r'./Model/Mask_model/learnTensorFlow-main/faceMaskDetection/face_mask.model'
p_protxt = r'./Model/Mask_model/learnTensorFlow-main/faceMaskDetection/deploy.prototxt.txt'
p_caffemodel = r'./Model/Mask_model/learnTensorFlow-main/faceMaskDetection/res10_300x300_ssd_iter_140000.caffemodel'
def get_project_path():
    return project_path

def get_frame_cap_path():
    return frame_cap_path

def get_assets_folder():
    return assets_folder

def get_exported_model_folder():
    return exported_model_folder

def get_facenet_model_path():
    return facenet_model_path

def get_full_model_path():
    return full_model_path

def get_half_model_path():
    return half_model_path

def get_full_face_classes_path():
    return full_face_classes_path

def get_half_face_classes_path():
    return half_face_classes_path

def get_mask_detect_model_folder():
    return mask_detect_model_folder

def get_p_model():
    return p_model

def get_p_protxt():
    return p_protxt

def get_p_caffemodel():
    return p_caffemodel