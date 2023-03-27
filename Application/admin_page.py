from time import sleep
import path.path_settings as ps
import flet as ft

#camera detection
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
from datetime import datetime
from mtcnn import MTCNN
import Model.face_processing as fp
import tensorflow as tf
import numpy as np
import PIL as pil
import pymongo
import joblib
import pickle
import cv2
import os

from flet import ListView, AlertDialog, FilePicker, FilePickerResultEvent, TextButton, ButtonStyle, VerticalDivider, Divider, margin, border_radius, Image,padding,alignment,Container,Dropdown, dropdown, Column, Ref, Text, theme,ElevatedButton,colors,IconButton, Page, Row, TextField, icons

import Database.database_manage as dbm
import Model.model_manage as mm
frame_num = 0
def admin_panel(page: Page): 

    ################################################## Camera ##################################################################
    def camera_on(run=False):
        if run == True:
            #Load threshold
            full_face_threshold,half_face_threshold = dbm.get_threshold()
            #Load model
            facenet_model = tf.keras.models.load_model(ps.get_facenet_model_path())
            file_half_model = ps.get_half_model_path()
            loaded_model_half = joblib.load(file_half_model)
            file_full_model = ps.get_full_model_path()
            loaded_model = pickle.load(open(file_full_model, 'rb'))
            out_encoder = LabelEncoder()
            out_encoder.classes_ = np.load(ps.get_classes_path())
            myclient = pymongo.MongoClient(r"mongodb+srv://parit:132456@cluster1.gktrrcl.mongodb.net/?retryWrites=true&w=majority")
            mydb = myclient["Face_recognition_project"]

            face_mask = ['Masked', 'No mask']
            size = 224

            # Load face detection and face mask model
            path = ps.get_mask_detect_model_folder()
            p_model = ps.get_p_model()
            
            #Mask model
            model = tf.keras.models.load_model(p_model)
            
            #cv2
            color = (0, 255, 0)
            cap = cv2.VideoCapture(0)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(os.path.join(path, 'test4.avi'),
                        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
            loading_camera_dialog.open = False
            page.update()
            while True:
                global frame_num
                ret, frame = cap.read()
                detector = MTCNN()
                faces  = detector.detect_faces(frame)
                for result in faces:

                    startX, startY, w, h = result['box']
                    endX, endY = startX + w, startY + h
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (size, size))
                    face = np.reshape(face, (1, size, size, 3)) / 255.0
                    result = np.argmax(model.predict(face))
        
                
                    if result == 0:                     #masked face
                        label = face_mask[result]
                        face_extract = mm.extract_cam_face(frame,(startX, startY, endX,endY))
                        cut_mask_img = fp.cut_mask_for_face_regconition(face_extract)
                        predict_result, person_prob= mm.predict_face_mask_face(cut_mask_img,facenet_model,loaded_model_half)
                        name_like_crimminal = str(out_encoder.inverse_transform(predict_result)[0])
                        print("name:",name_like_crimminal)
                        mycol = mydb[name_like_crimminal]
                        myquery = {}
                        mydoc = mycol.find_one(myquery)
                        face_arr = np.array(mydoc['face_image'])

                        cut_face_arr = fp.cut_mask_for_face_regconition(face_arr)
                        emd_face_arr = fp.get_embedding(facenet_model,cut_face_arr)
                        emd_face_mask = fp.get_embedding(facenet_model,cut_mask_img)
                        emd_face_arr = mm.l2_normalize(emd_face_arr)
                        emd_face_mask = mm.l2_normalize(emd_face_mask)

                        print("emd face mask: ",emd_face_mask.shape)
                        print("emd face arr: ",emd_face_arr.shape)
                        #use threshold
                        threshold_mask_face = float(half_face_threshold)
                        check_face = distance.euclidean(emd_face_mask,emd_face_arr)
                        print("check_face_distance:",check_face)
                        #print("Mask face distance: ",check_face)
                        if check_face < threshold_mask_face :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like '+ name_like_crimminal , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            detect_item = {}
                            detect_item['time'] = str(datetime.today().replace(microsecond=0))
                            detect_item['type'] = "mask face"
                            detect_item['name'] = name_like_crimminal
                            detect_item['probability'] = int(max(person_prob[0])*100)
                
                            print("Time: {}\nType: {}\nname: {}".format(detect_item['time'],detect_item['type'],detect_item['name']))
                            
                            sleep(1)
                            cv2.imwrite(r"./Application/assets/frame_cap/frame_no_{}.png".format(frame_num), frame)
                            detect_lv.controls.append(
                                Container(
                                    content = Row([
                                                    Image(src=r"/frame_cap/frame_no_{}.png".format(frame_num),width = 150, height = 150),
                                                    Text(value="Time: {}\nType: {} \nName: {} \nProbability: {}%".format(detect_item['time'],detect_item['type'],detect_item['name'],detect_item['probability']),size=20)
                                                    ]
                                                  )
                                )
                            )
                            
                            frame_num += 1
                            detect_lv.controls.append(
                                Divider()
                            )
                            page.update()
                            
                        else :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like Normal people ' , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:                                   #non-masked face
                        label = face_mask[result]
                        name, emd_full_face, person_prob = mm.predict_face(frame,(startX,startY,endX,endY),facenet_model,loaded_model)
                        name_like_crimminal = str(out_encoder.inverse_transform(name)[0])
                        mycol = mydb[name_like_crimminal]
                        myquery = {}
                        mydoc = mycol.find_one(myquery)
                        face_arr = np.array(mydoc['face_image'])

                        emd_face_arr = fp.get_embedding(facenet_model,face_arr)
                        emd_face_arr = mm.l2_normalize(emd_face_arr)
                        #use threshold
                        threshold_full_face = float(full_face_threshold)
                        check_face = distance.euclidean(emd_full_face,emd_face_arr)
                        print("Check face:",check_face)
                        if check_face < threshold_full_face :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like '+ name_like_crimminal , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            detect_item = {}
                            detect_item['time'] = str(datetime.today().replace(microsecond=0))
                            detect_item['type'] = "full face"
                            detect_item['name'] = name_like_crimminal
                            detect_item['probability'] = int(max(person_prob[0])*100)
                            print(detect_item)
                            print("Time: {}\nType: {} Probability: {}\nname: {}".format(detect_item['time'],detect_item['type'],detect_item['probability'],detect_item['name']))          
                            sleep(1)
                            cv2.imwrite(r"./Application/assets/frame_cap/frame_no_{}.png".format(frame_num), frame)

                            #append face
                            detect_lv.controls.append(
                                Container(
                                    content = Row([
                                                    Image(src=r"/frame_cap/frame_no_{}.png".format(frame_num),width = 150, height = 150),
                                                    Text(value="Time: {}\nType: {} \nName: {} \nProbability: {}%".format(detect_item['time'],detect_item['type'],detect_item['name'],detect_item['probability']),size=20),
                                                    ]
                                                  ),
                                ),
                            )
                            frame_num += 1
                            detect_lv.controls.append(
                                Divider()
                            )
                            page.update()
                            
                        else :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like Normal people ' , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if ret == True:
                    out.write(frame)
                else:
                    pass

                cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Video', 800, 600)
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
    ########################################################################################################################################################################

    ##################################################### File Picker ###################################################################
    # Pick files dialog
    def pick_files_result(e: FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()
    
    def pick_image_file_result(e: FilePickerResultEvent):
        selected_file_image.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        selected_file_image.update()
    
    # Save file dialog
    def save_file_result(e: FilePickerResultEvent):
        save_file_path.value = e.path if e.path else "Cancelled!"
        save_file_path.update()
    
    save_file_dialog = FilePicker(on_result=save_file_result)
    save_file_path = Text()
    
     # Open directory dialog
    def get_directory_result(e: FilePickerResultEvent):
        directory_path.value = e.path if e.path else "Cancelled!"
        directory_path.update()
    
    def get_model_directory_result(e: FilePickerResultEvent):
        model_directory_path.value = e.path if e.path else "Cancelled!"
        model_directory_path.update()

    def get_image_directory_result(e: FilePickerResultEvent):
        image_path.value = e.path if e.path else "Cancelled!"
        image_path.update()

    #directory dialog
    get_directory_dialog = FilePicker(on_result=get_directory_result)
    directory_path = Text("Select your directory",size=18)
    get_model_directory_dialog = FilePicker(on_result=get_model_directory_result)
    model_directory_path = Text("Select your model directory",size=18)

    get_image_directory_dialog = FilePicker(on_result=get_image_directory_result)
    image_path = Text("Select your image")

    #Pick files dialog
    pick_files_dialog = FilePicker(on_result=pick_files_result)
    selected_files = Text("Select Files")

    pick_image_file_dialog = FilePicker(on_result=pick_image_file_result)
    selected_file_image = Text("Select File Image")

    # hide all dialogs in overlay
    page.overlay.extend([pick_files_dialog, pick_image_file_dialog, save_file_dialog, get_directory_dialog, get_image_directory_dialog, get_model_directory_dialog])
    ##################################################################################################################################
    
    def clear_dir(e):
        directory_path.value = "Select your directory"
        directory_path.update()

    def clear_model_dir(e):
        model_directory_path.value = "Select your model directory"
        model_directory_path.update()

    def clear_image_profile_dir(e):
        selected_file_image.value = "Select File Image"
        selected_file_image.update()

    def clear_image(e):
        selected_files.value = ""
        selected_files.update()

    #right panel
    right_panel = Container(
        content=Column([
            
        ])
    )
    page.overlay.append(right_panel)
    
    #camera function
    def camera_detect(e):
        print("Enter camera detect")
        dbm.clear_frame_cap()
        page.dialog = loading_camera_dialog
        loading_camera_dialog.open = True
        page.update()  
        sleep(1)
        camera_on(True)

    #Test image functions 
    def test_image(e):
        print("selected image:",selected_file_image.value)     
        if selected_file_image.value != None and selected_file_image != "Select File Image":
            image_file = selected_file_image.value
            cv2.imwrite(r'Application/assets/test_register_image.png',cv2.cvtColor(fp.extract_face(image_file), cv2.COLOR_RGB2BGR))
            im = pil.Image.open(r'Application/assets/test_register_image.png')
            im.show()
           
        page.update()            

    def insert_model_upload_click(e):
        page.dialog = upload_dialog
        upload_dialog.open = True
        page.update()
        
    def upload_model(e):
        close_upload_dialog(e)
        print("This is folder path:", model_directory_path.value)
        if model_directory_path.value != None:
            dbm.upload_model(model_directory_path.value,fm_threshold.value,hm_threshold.value)
            page.dialog = upload_complete_dialog
            upload_complete_dialog.open = True
            page.update()
        else:
            print("Invalid directory")
        

    def criminal_delete(e):
        if criminal_dropdown_name != None:
            dbm.delete_criminal(criminal_dropdown_name.value)
            page.dialog = criminal_delete_dialog
            criminal_delete_dialog.open = True
            page.update()
            
    def contact_delete(e):
        if criminal_dropdown_contact != None:
            result = dbm.check_contact_with_criminal(criminal_dropdown_contact.value)
            if result == True:
                dbm.delete_contact(criminal_dropdown_contact.value)
                page.dialog = contact_delete_dialog
                contact_delete_dialog.open =True
                page.update()
            else:
                page.dialog = contact_delete_dlg_modal
                contact_delete_dlg_modal.open = True
                page.update()

    #Model functions
    def train_model_clicked(e):
        train_dlg_open(e)
     
    def confirm_train_model_clicked(e):
        train_dlg_close(e)
        page.update()
        if directory_path.value != None:
            page.dialog = loading_dialog
            loading_dialog.open = True
            page.update()
            full_model_result = mm.train_full_face_model(directory_path.value,get_arange_parameters(fm_min_gamma.value,fm_max_gamma.value,fm_step_gamma.value,fm_min_c.value,fm_max_c.value,fm_step_c.value))
            half_model_result = mm.train_half_face_model(directory_path.value,get_arange_parameters(hm_min_gamma.value,hm_max_gamma.value,hm_step_gamma.value,hm_min_c.value,hm_max_c.value,hm_step_c.value))
            
            full_model_trained_result_lv.controls.append(
                Column([
                    Text("Full Face Model",size=20),
                    Text("Confusion matrix",text_align='center'),
                    Container(content=
                    Text("{}".format(full_model_result['confusion_matrix'])),
                    alignment=alignment.center
                    ),
                    Text("Classification report",text_align='center'),
                    Text("{}".format(full_model_result['classification_report'])),
                    Text("Threshold"),
                    Text("C4.5"),
                    Text("Threshold: {}".format(full_model_result['c4_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(full_model_result['c4_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(full_model_result['c4_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(full_model_result['c4_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(full_model_result['c4_threshold']['Accuracy'])),
                    Text("\nSigma 2"),
                    Text("Threshold: {}".format(full_model_result['sigma2_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(full_model_result['sigma2_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(full_model_result['sigma2_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(full_model_result['sigma2_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(full_model_result['sigma2_threshold']['Accuracy'])),
                    Text("\nSigma 3"),
                    Text("Threshold: {}".format(full_model_result['sigma3_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(full_model_result['sigma3_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(full_model_result['sigma3_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(full_model_result['sigma3_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(full_model_result['sigma3_threshold']['Accuracy'])),
                ],),
            )

            half_model_trained_result_lv.controls.append(
                Column([
                    Text("Half Face Model",size=20),
                    Text("Confusion matrix",text_align='center'),
                    Container(content=
                    Text("{}".format(half_model_result['confusion_matrix'])),
                    alignment=alignment.center
                    ),
                    Text("Classification report",text_align='center'),
                    Text("{}".format(half_model_result['classification_report'])),
                    Text("Threshold"),
                    Text("C4.5"),
                    Text("Threshold: {}".format(half_model_result['c4_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(half_model_result['c4_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(half_model_result['c4_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(half_model_result['c4_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(half_model_result['c4_threshold']['Accuracy'])),
                    Text("\nSigma 2"),
                    Text("Threshold: {}".format(half_model_result['sigma2_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(half_model_result['sigma2_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(half_model_result['sigma2_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(half_model_result['sigma2_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(half_model_result['sigma2_threshold']['Accuracy'])),          
                    Text("\nSigma 3"),
                    Text("Threshold: {}".format(half_model_result['sigma3_threshold']['Threshold'])),
                    Text("Precision: {:.2f}%".format(half_model_result['sigma3_threshold']['Precision'])),
                    Text("Recall: {:.2f}%".format(half_model_result['sigma3_threshold']['Recall'])),
                    Text("F1: {:.2f}%".format(half_model_result['sigma3_threshold']['F1'])),
                    Text("Accuracy: {:.2f}%".format(half_model_result['sigma3_threshold']['Accuracy'])),
                ],)
            )
            result = Container(
                content = Row([
                    full_model_trained_result_lv,
                    VerticalDivider(),
                    half_model_trained_result_lv,
            ]),
            width=700
            
            )
            loading_dialog.open = False
            complete_train_dialog.content = result
            page.update()
            page.dialog = complete_train_dialog
            complete_train_dialog.open = True
            page.update()
        else:
            print("doesn't have directory")
        return 0
        

    def test_full_model(e):
        if directory_path.value != None:
            mm.test_full_image(selected_files.value,ps.get_full_model_path())
            complete_test_dialog.content = Text("Test complete")
            page.dialog = complete_test_dialog
            complete_test_dialog.open = True
        else:
            print("doesn't have directory")   
    
    def test_half_model(e):
        if directory_path.value != None:
            mm.test_half_image(selected_files.value,ps.get_half_model_path())
            complete_test_dialog.content = Text("Test complete")
            page.dialog = complete_test_dialog
            complete_test_dialog.open = True
        else:
            print("doesn't have directory")          

    #parameter functions
    def get_arange_parameters(min_gamma=0.1,max_gamma=100,step_gamma='x10',min_c=0.1,max_c=100,step_c='x10'):
        c_values,gamma_values = [0.1],[0.1]
        print("step_gamma: ",step_gamma)
        print("step_c: ",step_c)
        #multiply case
        if (step_gamma != '' and step_c != ''):
            if (step_gamma[0] in ['x'] or step_c[0] in ['x']):
                if step_gamma[0] in ['x']:
                    result_gamma_output = []
                    result = float(min_gamma)
                    print("result",result)
                    while result <= float(max_gamma):
                        print("Now result: ",result)
                        result_gamma_output.append(result)
                        result*=float(step_gamma[1:])
                    print("Gamma: ",result_gamma_output)
                    gamma_values = np.array(result_gamma_output)
                if step_c[0] in ['x']:
                    result_c_output = []
                    result = float(min_c)
                    while result <= float(max_gamma):
                        print("Now result: ",result)
                        result_c_output.append(result)
                        result*=float(step_c[1:])
                    print("C: ",result_c_output)
                    c_values = np.array(result_c_output)
                else:
                    print("Empty field")
            else:
                c_values = np.arange(min_c,max_c,step_c)
                gamma_values = np.arange(min_gamma,max_gamma,step_gamma)
            return {'c_values':list(c_values), 'gamma_values':list(gamma_values), 'kernel':['rbf', 'poly', 'sigmoid','linear']}
        else:
            print("You have to input 2 parameters.")
            return {'c_values':list(c_values), 'gamma_values':list(gamma_values), 'kernel':['rbf', 'poly', 'sigmoid','linear']}

#register criminal functions
    def add_criminal_clicked(e):
        print("add criminal processing...")
        print("Select file image:",selected_file_image)
        dbm.add_criminal_data(cri_fn.value,cri_ln.value,cri_CID.value,cri_nation.value,cri_age.value,cri_sex.value,cri_offense.value,selected_file_image.value,criminal_dropdown_contact.value,cri_announced_date.value)
        register_dialog.title = Text('Criminal added')
        register_dialog.title_padding=padding.only(top=22,left=54)
        page.dialog = register_dialog
        register_dialog.open = True
        print("Successful criminal added")
        page.update()

    def add_contact_clicked(e):
        print("add contact processing...")
        dbm.add_contact_data(contact_fn.value,contact_ln.value,contact_email.value,contact_num.value,contact_lo.value,contact_pas_code.value,contact_add_no.value,contact_vil_no.value,contact_road.value,contact_sub_dist.value,contact_dist.value,contact_provi.value)
        register_dialog.title = Text('Contact added')
        register_dialog.title_padding=padding.only(top=22,left=60)
        page.dialog = register_dialog
        register_dialog.open = True
        print("Successful contact added")
        page.update()

    def clear_criminal_clicked(e):
        cri_fn.value = ''
        cri_ln.value = ''
        cri_CID.value = ''
        cri_nation.value = ''
        cri_age.value = ''
        cri_sex.value =''
        cri_offense.value = ''
        cri_announced_date.value = ''
        criminal_dropdown_contact.value = ''
        page.update()

    def clear_contact_clicked(e):
        contact_fn.value = ''
        contact_ln.value = ''
        contact_email.value = ''
        contact_num.value = ''
        contact_lo.value = ''
        contact_pas_code.value = ''
        contact_add_no.value = ''
        contact_vil_no.value = ''
        contact_road.value = ''
        contact_sub_dist.value = ''
        contact_dist.value = '' 
        contact_provi.value = ''
        page.update()

    def add_criminal_register():
        criminal_dropdown_contact.options = get_contact()
        criminal_reg_lv.controls.append(
            Container(
                content=Column([
                Text("Criminal Information",size=30, weight="bold",),
                Row([
                    Image(src="/test_image.png",width=150,height=150),
                    Column([
                        Container(
                            content=Text("Criminal Profile Image",size=24,weight="w600"),
                            alignment=alignment.center,
                            margin=margin.only(top=10)
                        ),
                        
                        Container(
                            content= Column([
                                Container(
                                    content=selected_file_image,
                                    alignment=alignment.center
                            ),
                            Row([
                                ElevatedButton(
                                "Select Image",
                                icon=icons.FOLDER_OPEN,
                                on_click=lambda _: pick_image_file_dialog.pick_files(
                                        allow_multiple=False,
                                ),
                                disabled=page.web,
                                ),
                                ElevatedButton("Clear",
                                icon=icons.FORMAT_CLEAR,
                                on_click=clear_image_profile_dir
                                ), 
                                ElevatedButton(
                                "Test image",
                                icon=icons.UPLOAD,
                                on_click=test_image,
                                disabled=page.web,
                                ),
                            ])
                            ],
                            horizontal_alignment='center'
                            ),
                            alignment=alignment.center,
                            padding=padding.only(right=20,left=20,bottom=20)
                        ),
                        ], 
                        horizontal_alignment='center'
                    )
                ],
                alignment='center'

                ),
        
                Row([
                    Container(
                        content=cri_fn,
                        width = 260,
                    ),
                    Container(
                        content=cri_ln,
                        width= 260
                    ),
                    Container(
                        content=cri_announced_date,
                        width=160
                    ), 
                    ]),
                Row([
                    Container(
                        content=cri_CID,
                        width=465
                    ),
                    Container(
                        content=cri_nation,
                        width=225
                    )
                ]),
                
                Row([
                    Container(
                        content=cri_age,
                        width=230
                    ),
                    
                    Container(
                        content=cri_sex,
                        width=225
                    ),
                    Container(
                        content=criminal_dropdown_contact,
                        width=225
                    ),
                    ],
                ),
                cri_offense,
                Container(
                        content=Row([
                        ElevatedButton(
                            "Add Criminal",
                            icon=icons.PERSON_ADD_ALT_ROUNDED,
                            on_click=add_criminal_clicked,
                            disabled=page.web,
                        ),
                        ElevatedButton(
                            "Clear Criminal",
                            icon=icons.CLEAR_ROUNDED,
                            on_click=clear_criminal_clicked,
                            disabled=page.web,
                        ),
                    ],
                    alignment='center',
                    ),
                    alignment=alignment.center,
                ),
                Divider(),
                Text("Contact Information",size=30, weight="bold"),
                Row([
                    Container(
                        content=contact_fn,
                        width = 345,
                    ),
                    Container(
                        content=contact_ln,
                        width= 345
                    )    
                    ]),
                Row([
                    Container(
                        content=contact_email,
                        width= 345
                    ),
                    Container(
                        content=contact_num,
                        width= 345
                    ),
                ]),
                Row([
                    Container(
                        content=contact_lo,
                        width= 462
                    ),
                    Container(
                        content=contact_pas_code,
                        width=227
                    ),
                ]),
                Row([
                    Container(
                        content=contact_add_no,
                        width=226
                    ),
                    Container(
                        content=contact_vil_no,
                        width=226
                    ),
                    Container(
                        content= contact_road,
                        width=227
                    ),
                ]),
                Row([
                    Container(
                        content=contact_sub_dist,
                        width=226
                    ),
                    Container(
                        content=contact_dist,
                        width=226
                    ),
                    Container(
                        content=contact_provi,
                        width=227
                    ),
                ]),
                Row([
                    Container(
                        content=ElevatedButton(
                        "Add Contact",
                        icon=icons.PERSON_ADD_ALT_ROUNDED,
                        on_click=add_contact_clicked,
                        disabled=page.web,
                    ),
                    alignment=alignment.center,
                    ),
                    Container(
                        content=ElevatedButton(
                        "Clear Contact",
                        icon=icons.CLEAR_ROUNDED,
                        on_click=clear_contact_clicked,
                        disabled=page.web,
                    ),
                    alignment=alignment.center,
                    ),
                ],
                alignment= 'center',
                ),
                
            ]),
            )
        )
        page.update()

    def reset_criminal_register():
        criminal_reg_lv.controls = []

    def get_criminal_register():
        reset_criminal_register()
        add_criminal_register()
        return criminal_reg_lv

#Display functions
    def close_dlg(e):
        contact_delete_dlg_modal.open = False
        page.update()

    def close_upload_dialog(e):
        upload_dialog.open = False
        page.update()

    def close_complete_test_dialog(e):
        complete_test_dialog.open = False
        page.update()

    def close_register_dlg(e):
        register_dialog.open = False
        page.update()

    def close_fm_dlg(e):
        model_parameter_dialog.open = False
        page.update()

    def full_model_test_parameter_clicked(e):
        parameters = get_arange_parameters(fm_min_gamma.value,fm_max_gamma.value,fm_step_gamma.value,fm_min_c.value,fm_max_c.value,fm_step_c.value) 
        model_parameter_dialog.title = ft.Text("Full face model parameters")
        model_parameter_dialog.content = Container(
            Column([
                Row([
                    Text("Gamma: {}".format(parameters["gamma_values"]),)
                ]),
                Row([
                    Text("C: {}".format(parameters['c_values']),)
                ])
            ]),
            width=300,
            height=60,
            padding=10
        )
        page.dialog = model_parameter_dialog
        model_parameter_dialog.open = True
        page.update()

    def half_model_test_parameter_clicked(e):
        parameters = get_arange_parameters(hm_min_gamma.value,hm_max_gamma.value,hm_step_gamma.value,hm_min_c.value,hm_max_c.value,hm_step_c.value) 
        model_parameter_dialog.title = ft.Text("Half face model parameters")
        model_parameter_dialog.content = Container(
            Column([
                Row([
                    Text("Gamma: {}".format(parameters["gamma_values"]),)
                ]),
                Row([
                    Text("C: {}".format(parameters['c_values']),)
                ])
            ]),
            width=300,
            height=60,
            padding=10
        )
        page.dialog = model_parameter_dialog
        model_parameter_dialog.open = True
        page.update()

    def profile_clicked(e):
        for each_profile in profile_dialog:
            if each_profile['name'] == e.control.data:
                profile_detail = each_profile['profile_detail']
        print(profile_detail)
        page.dialog = profile_detail
        profile_detail.open = True
        page.update()
   
    def add_criminal():
        criminal_name = dbm.list_criminal()
        criminal_dropdown_name.options = []
        for i in range(1, len(criminal_name) + 1):
            if dbm.check_non_criminal_categories(criminal_name[i-1]):
                add_criminal_options(criminal_name[i-1])
                add_profile_dialog(criminal_name[i-1])
                criminal_box_lv = ft.ListView(expand=1, horizontal=True)
                criminal_box_lv.controls.append(
                    Container(
                                content=
                                    Row([
                                            
                                            Image(src="/criminal_images/{}.png".format(criminal_name[i-1]), width=100, height=90),
                                            Column([
                                                #Text(value=criminal_name[i-1],size=15),
                                                Container(
                                                    content= ElevatedButton(
                                                            "{}".format("Profile"),
                                                            #icon=icons.PERSON,
                                                            on_click= profile_clicked,
                                                            data = criminal_name[i-1],
                                                            disabled=page.web,
                                                            )
                                                    ,
                                                    #content=profile_button,
                                                    alignment=alignment.center,
                                                )
                                            ],
                                            alignment='center'
                                            )
                                            ], 
                                        ),
                                        margin = margin.only(left=10)
                            )
                )
                criminal_lv.controls.append(
                    Container(
                        content=Row([
                            criminal_box_lv
                                ]),
                        alignment=alignment.center,
                        width=150,
                        height=100,
                        bgcolor=colors.BLUE_100,
                        border_radius=border_radius.all(5),
                    )
                )
        
    def reset_criminal():
        criminal_lv.controls = []
    
    def get_criminal():
        reset_criminal()
        add_criminal()
        return criminal_lv
          
    def add_criminal_options(criminal_name):
        criminal_dropdown_name.options.append(dropdown.Option(criminal_name))          

    def add_profile_dialog(criminal_name):
        profile_cri_info_dialog_lv = ft.ListView(expand=1, spacing=20, padding=padding.only(left=10,right=0,top=10,bottom=10), auto_scroll=True)
        profile_cri_contact_dialog_lv = ft.ListView(expand=1, spacing=20, padding=padding.only(left=10,right=0,top=10,bottom=10), auto_scroll=True)
        if dbm.check_non_criminal_categories(criminal_name):
            profile_data = dbm.list_criminal_profile(criminal_name)
            profile_cri_info_dialog_lv.controls.append(ft.Column([
                                            ft.Image(src='/criminal_images/{}.png'.format(criminal_name),width=150,height=150),
                                            ft.Text("Criminal Infomation"),
                                            ft.Text("Name: {} {}".format(profile_data['criminal_first_name'],profile_data['criminal_last_name'])),
                                            ft.Text("Citizen ID: {}".format(profile_data['criminal_citizen_id'])),
                                            ft.Text("Nationality: {}".format(profile_data['criminal_nationality'])),
                                            ft.Text("Age: {} Sex: {}".format(profile_data['criminal_age'],profile_data['criminal_sex'])),
                                            ft.Text("Offense: \n{}".format(profile_data['criminal_offense'])),
                                        ]))
            profile_cri_contact_dialog_lv.controls.append(ft.Column([
                                            ft.Text("Contact Information"),
                                            ft.Text("Name: {}".format(profile_data['contact_first_name'],profile_data['contact_last_name'])),
                                            ft.Text("Email: {}".format(profile_data['contact_email'])),
                                            ft.Text("Number: {}".format(profile_data['contact_number'])),
                                            ft.Text("Location: {}".format(profile_data['contact_location'])),
                                            ft.Text("Postal Code: {}".format(profile_data['contact_pastal_code'])),
                                            ft.Text("Address No: {}".format(profile_data['contact_address_no'])),
                                            ft.Text("Village No: {}".format(profile_data['contact_village_no'])),
                                            ft.Text("Road: {}".format(profile_data['contact_road'])),
                                            ft.Text("Sub District: {}".format(profile_data['contact_sub_district'])),
                                            ft.Text("District: {}".format(profile_data['contact_district'])),
                                            ft.Text("Province: {}".format(profile_data['contact_province']))    
                                    ]))
            profile_dialog.append({"name":criminal_name,
                                   "profile_detail":
                                   ft.AlertDialog(
                                    title=Row([ft.Text("Criminal Profile"),ft.Text("Announced: {}".format(datetime.strptime(str(profile_data['criminal_anounced_date'])[:10], "%Y-%m-%d").strftime("%d/%m/%Y")))], spacing=100), on_dismiss=lambda e: print("Profile Dialog dismissed!"),
                                    content=Container(
                                        content=Row([
                                    profile_cri_info_dialog_lv,
                                    VerticalDivider(width=0),
                                    profile_cri_contact_dialog_lv,
                                ]),
                                border=ft.border.all(4,"#8CF0FF"),
                                border_radius=ft.border_radius.all(10),
                                width=300,
                                height=400,
                                padding=10
                                ),
                                content_padding=padding.only(top=15,right=20,left=20,bottom=2)
                            )
                        })

    def get_contact():
        contact_options = []
        for each_contact in dbm.list_contact():
            contact_options.append(add_contact_options(each_contact['id'],each_contact['name']))
        page.update()
        return contact_options
    
    def add_contact_options(contact_key,contact_name):
        return ft.dropdown.Option(key=contact_key,text=contact_name,)

    #Dialog Functions
    def complete_train_dialog_close(e):
        complete_train_dialog.open = False
        page.update()
             
    def train_dlg_close(e):
        train_dlg.open = False
        page.update()

    def train_dlg_open(e):
        page.dialog = train_dlg
        train_dlg.open = True
        page.update()

    ####  show right panel function  ####
    def camera_clicked(e):
        dbm.clear_frame_cap()
        mode.value = f"Camera"
        right_panel.content = Row([
                    #Camera detect detail
                    Container(
                        content=Column([
                            Container(
                                    content=Text("Detect Detail",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=15,right=20)
                                ),
                                Container(
                                    content=detect_lv,
                                    border_radius=border_radius.all(20),
                                    margin=margin.only(left=18),
                                    bgcolor="#D4F4FF",
                                    width=680,
                                    height=440,
                                )
                        ]),
                    alignment=alignment.center,
                    bgcolor=colors.WHITE,
                    width=720,
                    height=560,
                    border_radius=border_radius.all(20),
                    margin=margin.only(top=10,bottom=10,left=15)
                    ),
                    #Criminal list
                    Container(
                        content=Column([
                                Container(
                                    content=Text("Criminal List",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=15,right=20)
                                ),
                                
                                Container(    
                                    content=get_criminal(),
                                    height=380,
                                    alignment=alignment.center,
                                ),
                                Container(
                                    content=ElevatedButton(
                                        "Camera enable",
                                        icon=icons.CAMERA_ALT,
                                        on_click= camera_detect,
                                        disabled=page.web,
                                    ),
                                    alignment=alignment.center,
                                ),
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=275,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10)
                        ),
        ])
        right_panel.update()
        page.update()

###database click
    def database_clicked(e):
        dbm.clear_frame_cap()
        detect_lv.controls.clear()
        delete_lv.controls.clear()
        model_directory_path_lv.controls.clear()
        model_directory_path_lv.controls.append(model_directory_path)
        mode.value = f"Upload"
        delete_lv.controls.append(Column([
                                        Container(
                                            content=Text("Delete Criminal",size=24,weight="w600"),
                                            alignment=alignment.center,
                                            margin=margin.only(top=10)
                                    ),
                                    Container(
                                        content=Divider(),
                                        margin=margin.only(left=20,right=20)
                                    ),
                                    Container(
                                        content=criminal_dropdown_name,
                                        alignment=alignment.center,
                                        padding=padding.only(right=20,left=20),
                                        height=80
                                    ),
                                    Container(
                                        content=ElevatedButton("Delete",
                                        icon=icons.DELETE_OUTLINED,
                                        on_click=criminal_delete,
                                        disabled=page.web,),
                                        alignment=alignment.center,
                                        padding=padding.only(right=20,left=20)
                                    ),
                                    Container(
                                        content=Divider(),
                                        margin=margin.only(left=20,right=20)
                                    ),
                                    Container(
                                            content=Text("Delete Contact",size=24,weight="w600"),
                                            alignment=alignment.center,
                                            margin=margin.only(top=10)
                                    ),
                                    Container(
                                        content=criminal_dropdown_contact,
                                        alignment=alignment.center,
                                        padding=padding.only(right=20,left=20),
                                        height=80
                                    ),
                                    Container(
                                        content=ElevatedButton("Delete",
                                        icon=icons.DELETE_OUTLINED,
                                        on_click=contact_delete,
                                        disabled=page.web,),
                                        alignment=alignment.center,
                                        padding=padding.only(right=20,left=20)
                                    ),
                                    ]))
        right_panel.content = Row([
                    #Criminal Information
                    #Insert Criminal
                    Container(
                        content=
                        Column([
                            Container(
                            content=Text("Register Criminal",size=24,weight="w600"),
                            alignment=alignment.center,
                            margin=margin.only(top=10)
                            ),
                            Container(
                                content=Divider(),
                                margin=margin.only(left=20,right=20)
                            ),
                            Container(
                            content=get_criminal_register(),
                            height=450,
                            alignment=alignment.center,
                        ),
                        ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=740,
                        height=600,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,left=10,bottom=10)
                        ),
                    Column([
                                Container(
                                    content=delete_lv,
                                    bgcolor=colors.WHITE,
                                    width=260,
                                    height=240,
                                    border_radius=border_radius.all(20),
                                    margin=margin.only(top=10)      
                                ),
                                Container(
                                        content=Column([
                                            Container(
                                                content=Text("Insert Models",size=24,weight="w600"),
                                                alignment=alignment.center,
                                                margin=margin.only(top=10)
                                            ),
                                            Container(
                                                content=Divider(),
                                                margin=margin.only(left=20,right=20)
                                            ),
                                            model_directory_path_lv,
                                            Container(
                                                content=Divider(),
                                                margin=margin.only(left=20,right=20)
                                            ),
                                            Container(
                                                content=ElevatedButton(
                                                "Select directory",
                                                icon=icons.FOLDER_OPEN,
                                                on_click=lambda _: get_model_directory_dialog.get_directory_path(),
                                                disabled=page.web,
                                            ),
                                            alignment=alignment.center,
                                            ),
                                        

                                            Container(
                                                content=ElevatedButton(
                                                "Insert",
                                                icon=icons.UPLOAD,
                                                on_click=insert_model_upload_click,
                                                disabled=page.web,
                                            ),
                                                alignment=alignment.center
                                            ),
                                            Container(
                                                content=ElevatedButton(
                                                "Clear",
                                                icon=icons.FORMAT_CLEAR,
                                                on_click=clear_model_dir
                                            ),
                                                alignment=alignment.center
                                            )
                                            
                                        
                                        ]),
                                    bgcolor=colors.WHITE,
                                    width=260,
                                    height=290,
                                    border_radius=border_radius.all(20),
                                    margin=margin.only(bottom=10) 
                                )
                                ])
                ])
        right_panel.update()
        page.update()
    
    def model_clicked(e):
        param_lv.controls.clear()
        mode.value = f"Model"
        param_lv.controls.append(
            Container(
                content = Column([
                    Container(
                        content=Column([
                            Container(content=Text("Full face adjustment",size=22), margin=margin.only(bottom=10)),
                            Row([Text("Gamma value:",size=18),
                            Container(
                                content=fm_min_gamma,
                                width=105
                            ),
                            Container(
                                content=fm_max_gamma,
                                width=110
                            ),
                            Container(
                                content=fm_step_gamma,
                                width=90
                            ),
                            ]),
                            Row([Text("C value:",size=18),
                            Container(
                                content=fm_min_c,
                                padding=padding.only(left=52),
                                width=158
                            ),
                            Container(
                                content=fm_max_c,
                                width=110,
                            ),
                            Container(
                                content=fm_step_c,
                                width=90,
                            ),
                            ]),
                            Container(
                                content=ft.ElevatedButton(text="Test Parameters", on_click=full_model_test_parameter_clicked),
                                alignment=ft.alignment.center
                            )
                        ],
                        alignment = "center"
                        ), 
                        margin = margin.only(bottom=20)          
                    ),
                     Container(
                        content=Column([
                            Container(Text("Half face adjustment",size=22), margin=margin.only(bottom=10)),
                            Row([Text("Gamma value:",size=18),
                            Container(
                                content=hm_min_gamma,
                                width=105
                            ),
                            Container(
                                content=hm_max_gamma,
                                width=110
                            ),
                            Container(
                                content=hm_step_gamma,
                                width=90
                            ),
                            ]),
                            Row([Text("C value:",size=18),
                            Container(
                                content=hm_min_c,
                                padding=padding.only(left=52),
                                width=158
                            ),
                            Container(
                                content=hm_max_c,
                                width=110,
                            ),
                            Container(
                                content=hm_step_c,
                                width=90,
                            )]),
                            Container(
                                content=ft.ElevatedButton(text="Test Parameters", on_click=half_model_test_parameter_clicked),
                                alignment=ft.alignment.center
                            ),
                        ],
                        alignment = "center"
                        ),           
                    ),

                ])
            )
        )
        right_panel.content = Row([
            Container(
                content = Column([
                     Container(
                        content=Column([
                            #Train model
                                Container(
                                    content=Text("Train Model",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=20,right=20)
                                ),
                                Container(    
                                    content=Column([  
                                        Container(    
                                            content=Column([
                                                Container(content=
                                                directory_path,
                                                alignment=alignment.center,
                                                margin=margin.only(right=10,left=15)
                                                )
                                            ])
                                        ),
                                        Container(
                                        content=Divider(),
                                        margin=margin.only(left=20,right=20)
                                        ),
                                        Row([
                                        Container(
                                            content=ElevatedButton("Select Folder",
                                            icon=icons.FOLDER_OPEN,
                                            on_click=lambda _: get_directory_dialog.get_directory_path(),
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20,left=20)
                                        ),
                                        Container(
                                            content=ElevatedButton("Clear",
                                            icon=icons.FORMAT_CLEAR,
                                            on_click=clear_dir,
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20)
                                        ),
                                        Container(
                                            content=ElevatedButton("Train",
                                            icon=icons.MODEL_TRAINING,
                                            on_click=train_model_clicked,
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20,left=20)
                                        )    
                                        ]),
                                    ]),
                                    alignment=alignment.center,
                                ),
    
                            ]),
                        bgcolor=colors.WHITE,
                        width=490,
                        height=215,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=15,bottom=5,left=15)
                        ),
                    Container(
                        content=Column([
                            #Test model
                                Container(
                                    content=Text("Test Model",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=20,right=20)
                                ),
                                Container(    
                                    content=Column([
                                        Container(content=
                                                selected_files,
                                                alignment=alignment.center,
                                                margin=margin.only(right=10,left=15)
                                        ),
                                        Container(
                                        content=Divider(),
                                        margin=margin.only(left=20,right=20)
                                        ),
                                    Row([
                                        Container(
                                        content=ElevatedButton("Select image",
                                        icon=icons.IMAGE,
                                        on_click=lambda _: pick_files_dialog.pick_files(
                                        allow_multiple=False,
                                        ),
                                        disabled=page.web,),
                                        alignment=alignment.center,
                                        padding=padding.only(right=20,left=100)
                                        ),
                                        Container(
                                        content=ElevatedButton("Clear",
                                        icon=icons.FORMAT_CLEAR,
                                        on_click=clear_image,
                                        disabled=page.web,),
                                        alignment=alignment.center,
                                        padding=padding.only(right=20)
                                        ),   
                                        ],
                                        ),
                                    Column([
                                        Container(
                                            content=ElevatedButton("Test with Full Face Model",
                                            icon=icons.MODEL_TRAINING,
                                            on_click=test_full_model,
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20,left=20)
                                        ),   
                                        Container(
                                            content=ElevatedButton("Test with Half Face Model",
                                            icon=icons.MODEL_TRAINING,
                                            on_click=test_half_model,
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20,left=20)
                                        ) 
                                        ])
                                    ]),
                                    alignment=alignment.center,
                                ),
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=490,
                        height=300,
                        border_radius=border_radius.all(20),
                        margin=margin.only(bottom=10,left=15)
                        ),

                ])
            ),
            #Parameter tuning
            Container(
                content = Container(
                        content=Column([
                                    Container(
                                        content=Text("Parameter tuning",size=24,weight="w600"),
                                        alignment=alignment.center,
                                        margin=margin.only(top=10)
                                        ),
                                    Container(
                                        content=Divider(),
                                        margin=margin.only(left=20,right=20)
                                        ),
                                    param_lv,
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=500,
                        height=600,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=15,right=15,bottom=15)
                        ),
            )
            ])
        right_panel.update()
        page.update()
### 
    def exit_clicked(e):
        page.window_destroy()
        return 0
        
    #Global settings
    page.title = "Face Recognitioni Scanner"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.window_center()
    page.bgcolor = "#D8FFFD"
    
    #Global variable

        #criminal information
    cri_fn=TextField(label="First name")
    cri_ln=TextField(label="Last name")
    cri_CID = TextField(label="Citizen ID")
    cri_nation = TextField(label="Nationality")
    cri_age = TextField(label="Age")
    cri_sex = Dropdown(
                           label="Sex",
                            width=100,
                            options=[
                                ft.dropdown.Option("Male"),
                                ft.dropdown.Option("Female"),
                            ])
    cri_offense = TextField(
                        label="Offense",
                        multiline=True,
                        min_lines=1,
                        max_lines=3,
                    )
    criminal_dropdown_contact = Dropdown(label="Contact Person",
                options=get_contact()
                )
    cri_announced_date = TextField(label="Announcement Date",hint_text='dd/mm/yyyy format')
        #contact information
    contact_fn = TextField(label="First name")
    contact_ln = TextField(label="Last name")
    contact_email = TextField(label="Email")
    contact_num = TextField(label="Number")
    contact_lo = TextField(
                                label="location",
                                multiline=True,
                                min_lines=1,
                                max_lines=3,
                            )
    contact_pas_code = TextField(label="Postal Code")
    contact_add_no = TextField(
                                label="Address No."
                            )
    contact_vil_no = TextField(
                                label="Village No."
                            )
    contact_road = TextField(
                                label="Road"
                            )
    contact_sub_dist = TextField(label="Sub-district")
    contact_dist = TextField(label="District") 
    contact_provi = TextField(label="Province")
        #parameters
    fm_min_gamma = TextField(label="Gamma minimum",text_size=15)
    fm_max_gamma = TextField(label="Gamma maximum",text_size=15)
    fm_step_gamma = TextField(label="Gamma step",text_size=15)
    fm_min_c = TextField(label="C minimum",text_size=15)
    fm_max_c = TextField(label="C maximum",text_size=15)
    fm_step_c = TextField(label="C step",text_size=15)
    hm_min_gamma = TextField(label="Gamma minimum",text_size=15)
    hm_max_gamma = TextField(label="Gamma maximum",text_size=15)
    hm_step_gamma = TextField(label="Gamma step",text_size=15)
    hm_min_c = TextField(label="C minimum",text_size=15)
    hm_max_c = TextField(label="C maximum",text_size=15)
    hm_step_c = TextField(label="C step",text_size=15)
        #threshold
    fm_threshold = TextField(label="Full face threshold")
    hm_threshold = TextField(label="Half face threshold")


        #dialog
    profile_dialog = []
    upload_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Please input your threshold"),
        content=Container(
            content=Column([
                Text("Full Face Threshold"),
                fm_threshold,
                Text("Half Face Threshold"),
                hm_threshold
            ]),
            height=200
        ),
        actions=[
            ft.TextButton("Upload", on_click=upload_model),
            ft.TextButton("Cancel", on_click=close_upload_dialog),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    upload_complete_dialog = ft.AlertDialog(
        modal=False,
        title=ft.Text("Upload complete"),
        title_padding=padding.only(top=24,left=50),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    loading_dialog = ft.AlertDialog(
        title=Column([
            Row([ft.ProgressRing(),Text("Training\nPlease wait...")],spacing=20),
        ],
        alignment='center'
        ),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    loading_dialog.title_padding = padding.only(top=26,left=35)

    loading_camera_dialog = ft.AlertDialog(
        title=Column([
            Row([ft.ProgressRing(),Text("Loading Camera\nPlease wait...")],spacing=20),
        ],
        alignment='center'
        ),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    loading_camera_dialog.title_padding = padding.only(top=26,left=40)

    train_dlg =  ft.AlertDialog(
        modal=True,
        title=ft.Text("Please confirm"),
        content=ft.Text("Don't forget to check or fill in parameters for training, if you don't we will train by default parameter values."),
        actions=[
            ft.TextButton("Train", on_click=confirm_train_model_clicked),
            ft.TextButton("Modify", on_click=train_dlg_close),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    complete_test_dialog =  ft.AlertDialog(
        modal=True,
        title=ft.Text("Test Succesfully"),
        content=ft.Text("",size=18),
        actions=[
            ft.TextButton("Yes",on_click=close_complete_test_dialog),
        ],
        actions_alignment='end',
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    contact_delete_dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("Deletation Fail"),
        content=ft.Text("Your contact have relations with criminal. Please remove your criminal first.",size=18),
        actions=[
            ft.TextButton("Yes",on_click=close_dlg),
        ],
        actions_alignment='end',
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    model_parameter_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Full face model parameters"),
        content=Column([
            ft.Text(),
        ]),
        actions=[
            ft.TextButton("Yes",on_click=close_fm_dlg)
        ],
        actions_alignment='end',
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    register_dialog = ft.AlertDialog(
        title=ft.Text("Register criminals & contacts",text_align='center'),
        title_padding=padding.only(top=23,left=54),
        on_dismiss=lambda e: print("dialog dismissed!"),
    )
    criminal_delete_dialog = ft.AlertDialog(
        title=ft.Text("Criminal deleted",text_align='center'),
        title_padding=padding.only(top=23,left=0),
        on_dismiss=lambda e: print("dialog dismissed!"),
    )
    contact_delete_dialog = ft.AlertDialog(
        title=ft.Text("Contact deleted",text_align='center'),
        title_padding=padding.only(top=23,left=0),
        on_dismiss=lambda e: print("dialog dismissed!"),
    )

        #list view
    model_directory_path_lv = ListView(expand=1, padding=padding.only(left=20), auto_scroll=True,horizontal=True)
    full_model_trained_result_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
    half_model_trained_result_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
    delete_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
    detect_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    criminal_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    criminal_reg_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
    param_lv = ListView(expand=1, spacing=10, padding=20)

    mode = Text(value="Admin Panel",size=40)
    criminal_dropdown_name = Dropdown(label="Criminal Name")
    complete_train_dialog = AlertDialog(
        modal=True,
        title=Text("Train complete"),
        content=Text("each model score"),
        actions=[
            TextButton("Ok", on_click=complete_train_dialog_close),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),

    )
    
    right_panel.content = Row([
                    #Camera detect detail
                    Container(
                        content=Column([
                            Container(
                                    content=Text("Detect Detail",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=15,right=20)
                                ),
                                Container(    
                                    content=Column(
                                        
                                    ),
                                    alignment=alignment.center,
                                )
                        ]),
                    alignment=alignment.center,
                    bgcolor=colors.WHITE,
                    width=740,
                    height=560,
                    border_radius=border_radius.all(20),
                    margin=margin.only(top=10,bottom=10,left=15)
                    ),
                    #Criminal list
                    Container(
                        content=Column([
                                Container(
                                    content=Text("Criminal List",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                Container(
                                    content=Divider(),
                                    margin=margin.only(left=15,right=20)
                                ),
                                Container(    
                                    content= get_criminal(),
                                    height=380,
                                    alignment=alignment.center,
                                ),
                                Container(
                                    content=ElevatedButton(
                                        "Camera enable",
                                        icon=icons.CAMERA_ALT,
                                        on_click= camera_detect,
                                        disabled=page.web,
                                    ),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=250,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10)
                        ),
        ])
    
    page.add(
        Column([
            Row([
                Container(
                    content=mode,
                    bgcolor="#D8FFFD"
                )
                ], alignment="center"
                ),
            Divider(), 
            Row([
                Container(content=
                    Column([
                        Container(content=
                                  Column([
                                    Image(src="/admin.png", width=100, height=100),
                                    Text(value="Admin",weight="w400",size=30),
                                    Divider(thickness=1, color="white")
                                    ],
                                    horizontal_alignment ="center"
                                    ),
                                padding=padding.only(top=10),
                                margin=margin.only(bottom=50),
                                alignment=alignment.center
                                ),
                        Container(content=
                                  Column([
                                    TextButton(content=Text("Camera", color=colors.BLACK, size=20),
                                        on_click=camera_clicked,
                                        ),
                                    TextButton(content=Text("Upload", color=colors.BLACK, size=20),
                                        on_click=database_clicked,
                                        ),
                                    TextButton(content=Text("Model", color=colors.BLACK, size=20),
                                        on_click=model_clicked
                                        ),
                                    ],
                                    horizontal_alignment ="center"
                                    ),
                                margin=margin.only(bottom=80),
                                alignment=alignment.center
                                ),
                        Container(content=
                                  Column([
                                    TextButton(content=Text("Exit", color=colors.BLACK, size=15), on_click=exit_clicked)
                                    ],
                                    horizontal_alignment ="center"
                                    ),
                                alignment=alignment.center
                                )
                    ]),
                    padding = padding.all(10),
                    bgcolor="#69A5FF",
                    width=200,
                    height=560,
                    alignment=alignment.center,
                    border_radius=border_radius.all(20)),
                Container(
                        content=right_panel,
                        bgcolor="#8DF1FF",
                        width=1030,
                        height=560,
                        border_radius=border_radius.all(20)
                        )
               ],
            )
        ])
    )

#comment this when run index