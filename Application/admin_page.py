import sys
sys.path.insert(0, '/Face_recognition_Project/Model/')
sys.path.insert(0, '/Face_recognition_Project/Model')
sys.path.insert(0, '/Face_recognition_Project/Database')
sys.path.insert(0, '/Face_recognition_Project/')

import flet
from time import sleep

#camera detection
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
from datetime import datetime
import face_processing as fp
import path_settings as ps
import tensorflow as tf
import numpy as np
import pymongo
import joblib
import pickle
import cv2
import os

from flet import ListView, AlertDialog, FilePicker, FilePickerResultEvent, TextButton, ButtonStyle, VerticalDivider, Divider, margin, border_radius, Image,padding,alignment,Container,Dropdown, dropdown, Column, Ref, Text, theme,ElevatedButton,colors,IconButton, Page, Row, TextField, icons

import camera_face_detect as cfm
import database_manage as dbm
import model_manage as mm
import path_settings as ps
def admin_panel(page: Page): 
    
    ##################################################### File Picker ###################################################################
    # Pick files dialog
    def pick_files_result(e: FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()

    pick_files_dialog = FilePicker(on_result=pick_files_result)
    selected_files = Text()
    selected_files = Text("Select your image",size=18)
    
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
    
    get_directory_dialog = FilePicker(on_result=get_directory_result)
    directory_path = Text("Select your directory",size=18)


    # hide all dialogs in overlay
    page.overlay.extend([pick_files_dialog, save_file_dialog, get_directory_dialog])
    ##################################################################################################################################
    
    ################################################## Camera ##################################################################
    def camera_on(run=False):
        
        if run == True:
            facenet_model = tf.keras.models.load_model(ps.get_facenet_model_path())
            file_half_model = ps.get_half_model_path()
            loaded_model_half = joblib.load(file_half_model)
            #loaded_model_half = pickle.load(open(file_half_model, 'rb'))
            file_full_model = ps.get_full_model_path()
            loaded_model = pickle.load(open(file_full_model, 'rb'))
            out_encoder = LabelEncoder()
            out_encoder.classes_ = np.load(ps.get_classes_path())
            myclient = pymongo.MongoClient("mongodb://localhost:27017/")
            mydb = myclient["Face_recognition_project"]

            face_mask = ['Masked', 'No mask']
            size = 224

            # Load face detection and face mask model
            path = ps.get_mask_detect_model_folder()
            p_model = ps.get_p_model()
            p_protxt = ps.get_p_protxt()
            p_caffemodel = ps.get_p_caffemodel()

            #Mask model
            model = tf.keras.models.load_model(p_model)
            faceNet = cv2.dnn.readNet(p_protxt,p_caffemodel)

            #detection info
            detection_info = []

            #cv2
            color = (0, 255, 0)
            cap = cv2.VideoCapture(0)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(os.path.join(path, 'test4.avi'),
                        cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
            while True:
                ret, frame = cap.read()
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                faceNet.setInput(blob)
                detections = faceNet.forward()

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence < 0.5:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (size, size))
                    face = np.reshape(face, (1, size, size, 3)) / 255.0
                    result = np.argmax(model.predict(face))
                
                    if result == 0:
                        label = face_mask[result]
                        face_extract = mm.extract_cam_face(frame,(startX, startY, endX,endY))
                        cut_mask_img = fp.cut_mask_for_face_regconition(face_extract)
                        emd_face_mask = fp.get_embedding(facenet_model,cut_mask_img)
                        name_like_crimminal = str(out_encoder.inverse_transform(mm.predict_face_mask_face(cut_mask_img,facenet_model,loaded_model_half))[0])
                        mycol = mydb[name_like_crimminal]
                        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
                        myquery = { "name": name_like_crimminal }
                        mydoc = mycol.find_one(myquery)
                        face_arr = np.array(mydoc['face_image'])
                        cut_face_arr = fp.cut_mask_for_face_regconition(face_arr)
                        emd_face_arr = fp.get_embedding(facenet_model,cut_face_arr)
                        emd_face_arr = mm.l2_normalize(emd_face_arr)
                      
                        #use threshold
                        threshold_mask_face = 1.5271
                        check_face = distance.euclidean(emd_face_mask,emd_face_arr)

                        #print("Mask face distance: ",check_face)
                        if check_face < threshold_mask_face :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like '+ name_like_crimminal , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            detect_item = {}
                            detect_item['time'] = str(datetime.today().replace(microsecond=0))
                            detect_item['type'] = "mask face"
                            detect_item['name'] = name_like_crimminal
                            #cv2.imwrite('my_video_frame.png', frame)
                            #print(detect_item)
                            print("Time: {} Type: {} name: {}".format(detect_item['time'],detect_item['type'],detect_item['name']))
                            
                            sleep(1)
                            detect_lv.controls.append(
                                Container(
                                    content = Row([
                                                    Image(src=f"https://cdn-icons-png.flaticon.com/512/1946/1946429.png",width = 150, height = 150),
                                                    Text("Time: {} Type: {} name: {}".format(detect_item['time'],detect_item['type'],detect_item['name']))
                                                    ]
                                                  )
                                )
                            )
                            page.update()
                            
                        else :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like Normal people ' , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        label = face_mask[result]
                        name, emd_full_face = mm.predict_face(frame,(startX,startY,endX,endY),facenet_model,loaded_model)
                        name_like_crimminal = str(out_encoder.inverse_transform(name)[0])
                        mycol = mydb[name_like_crimminal]
                        myquery = { "name": name_like_crimminal }
                        mydoc = mycol.find_one(myquery)
                        face_arr = np.array(mydoc['face_image'])
                        emd_face_arr = fp.get_embedding(facenet_model,face_arr)
                        emd_face_arr = mm.l2_normalize(emd_face_arr)
                        #use threshold
                        threshold_full_face = 1.3313
                        check_face = distance.euclidean(emd_full_face,emd_face_arr)
                        #print("Face arr: ",face_arr.shape)
                        print("Check face:",check_face)
                        if check_face < threshold_full_face :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like '+ name_like_crimminal , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            detect_item = {}
                            detect_item['time'] = str(datetime.today().replace(microsecond=0))
                            detect_item['type'] = "full face"
                            detect_item['name'] = name_like_crimminal
                            print(detect_item)
                            print("Time: {} Type: {} name: {}".format(detect_item['time'],detect_item['type'],detect_item['name']))
                            
                            sleep(1)
                            #cv2.imwrite("D:/Face_recognition_Project/Database/cache_images/frame%d.jpg" % count, frame)
                            detect_lv.controls.append(
                                Container(
                                    content = Row([
                                                    Image(src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png",width = 150, height = 150),
                                                    Text("Time: {} Type: {} name: {}".format(detect_item['time'],detect_item['type'],detect_item['name']))
                                                    ]
                                                  )
                                )
                            )
                            page.update()
                            
                        else :
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                            cv2.putText(frame, label + ' like Normal people ' , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                #cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
                #cv2.putText(frame, label , (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

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
    def clear_dir(e):
        directory_path.value = ""
        directory_path.update()
    
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
        camera_on(True)
        
        
    #database function
    def get_criminal_name():
        criminal_name = dbm.list_criminal()
        criminals = []
        criminal_dropdown_name.options = []
        for i in range(1, len(criminal_name) + 1):
            add_criminal_options(criminal_name[i-1])
            criminals.append(
                Container(
                    content=Text(value=criminal_name[i-1]),
                    alignment=alignment.center,
                    width=100,
                    height=50,
                    bgcolor=colors.BLUE_100,
                    border_radius=border_radius.all(5),
                )
            )
        return criminals
    
    def add_criminal_options(criminal_name):
        criminal_dropdown_name.options.append(dropdown.Option(criminal_name))
        
    def criminal_upload(e):
        print("This is folder path:", directory_path.value)
        if directory_path.value != None:
            #dbm.upload_full_face_criminal(directory_path.value)
            #dbm.upload_half_face_criminal(directory_path.value)
            dbm.upload_face_criminal(directory_path.value)
            #dbm.upload_model(directory_path.value)
            get_criminal_name()
            right_panel.update()
            #page.update()
    def model_upload(e):
        print("This is folder path:", directory_path.value)
        if directory_path.value != None:
            dbm.upload_model(directory_path.value)
    def criminal_delete(e):
        if criminal_dropdown_name != None:
            dbm.delete_criminal(criminal_dropdown_name.value)
      
    #Model functions
    def train_model(e):
        if directory_path.value != None:
            mm.train_full_face_model(directory_path.value)
            mm.train_half_face_model(directory_path.value)
            open_dlg_modal(e)
        else:
            print("doesn't have directory")
     
    def test_full_model(e):
        if directory_path.value != None:
            mm.test_full_image(selected_files.value,ps.get_full_model_path())
            complete_dialog.content = Text("Test complete")
            open_dlg_modal(e)
        else:
            print("doesn't have directory")   
    
    def test_half_model(e):
        if directory_path.value != None:
            mm.test_half_image(selected_files.value,ps.get_half_model_path())
            complete_dialog.content = Text("Test complete")
            open_dlg_modal(e)
        else:
            print("doesn't have directory")          
            
    #dialog
    def close_dlg(e):
        complete_dialog.open = False
        page.update()
     
    def open_dlg_modal(e):
        page.dialog = complete_dialog
        complete_dialog.open = True
        page.update()
             
    ####  show right panel function  ####
    def camera_clicked(e):
        # page.overlay.append(
        # Container(
        # content=detect_lv,
        # border_radius=border_radius.all(20),
        # margin=margin.only(top=200,left=260,bottom=10),
        # #bgcolor="#D4F4FF",
        # width=690,
        # height=440,
        # )
        # )
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
                                    content=Column(
                                        controls = get_criminal_name()
                                    ),
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
                        width=250,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10)
                        ),
        ])
        right_panel.update()
        page.update()
###       
    def  database_clicked(e):
        mode.value = f"Upload"
        right_panel.content = Row([
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
                                    margin=margin.only(left=20,right=20)
                                ),
                                Container(    
                                    content=Column(
                                        controls = get_criminal_name()
                                    ),
                                    alignment=alignment.center,
                                ),
    
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=300,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10,left=15)
                        ),
                    
                    #Insert Criminal
                    Container(
                        content = Column([
                                    Container(
                                    content=Text("Insert Criminal Profile",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                    Container(
                                    content=Divider(),
                                    margin=margin.only(left=20,right=20)
                                ),
                                    Container(
                                    content= Column([
                                            Container(
                                                content=directory_path,
                                                alignment=alignment.center
                                            ),
                                        Row([
                                            ElevatedButton(
                                                "Select directory",
                                                icon=icons.FOLDER_OPEN,
                                                on_click=lambda _: get_directory_dialog.get_directory_path(),
                                                disabled=page.web,
                                            ),
                                            ElevatedButton("Clear",
                                            icon=icons.FORMAT_CLEAR,
                                            on_click=clear_dir
                                            ),
                                        ]), 
                                            ElevatedButton(
                                                "Insert",
                                                icon=icons.UPLOAD,
                                                on_click=criminal_upload,
                                                disabled=page.web,
                                            ),
                                    ]),
                                    alignment=alignment.center,
                                    padding=padding.only(right=20,left=20)
                                ),
                                Container(
                                    content=Text("Insert Models",size=24,weight="w600"),
                                    alignment=alignment.center,
                                    margin=margin.only(top=10)
                                ),
                                    Container(
                                    content=Divider(),
                                    margin=margin.only(left=20,right=20)
                                ),
                                    Container(
                                    content= Column([
                                            Container(
                                                content=directory_path,
                                                alignment=alignment.center
                                            ),
                                            Container(
                                            content=ElevatedButton(
                                                "Select directory",
                                                icon=icons.FOLDER_OPEN,
                                                on_click=lambda _: get_directory_dialog.get_directory_path(),
                                                disabled=page.web,
                                            ),
                                            alignment=alignment.center,
                                            ),
                                            Container(
                                                content=ElevatedButton(
                                                "Insert",
                                                icon=icons.UPLOAD,
                                                on_click=model_upload,
                                                disabled=page.web,
                                            ),
                                                alignment=alignment.center
                                            ),
                                            Container(
                                                content=ElevatedButton(
                                                "Clear",
                                                icon=icons.FORMAT_CLEAR,
                                                on_click=clear_dir
                                            ),
                                                alignment=alignment.center
                                            )
                                            ])
                                )
                        
                        ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=340,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10)
                    ),
                    
                    #Delete Criminal
                    Container(
                        content = Column([
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
                                    padding=padding.only(right=20,left=20)
                                ),
                                    Container(
                                    content=ElevatedButton("Delete",
                                    icon=icons.DELETE_OUTLINED,
                                    on_click=criminal_delete,
                                    disabled=page.web,),
                                    alignment=alignment.center,
                                    padding=padding.only(right=20,left=20)
                                ),
                        ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=320,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10)
                    ),
                ])
        right_panel.update()
        page.update()
    
    def model_clicked(e):
        mode.value = f"Model"
        right_panel.content = Row([
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
                                        )])),
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
                                            on_click=train_model,
                                            disabled=page.web,),
                                            alignment=alignment.center,
                                            padding=padding.only(right=20,left=20)
                                        )    
                                        ]),
                                    ]),
                                    alignment=alignment.center,
                                ),
    
                            ]),
                        alignment=alignment.center,
                        bgcolor=colors.WHITE,
                        width=490,
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,right=10,bottom=10,left=15)
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
                        height=560,
                        border_radius=border_radius.all(20),
                        margin=margin.only(top=10,bottom=10)
                        ),
            ])
        right_panel.update()
        page.update()
### 
        
    #Global settings
    page.title = "Face Recognitioni Scanner"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.window_center()
    page.bgcolor = "#D8FFFD"
    
    #Global variable
    detect_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    mode = Text(value="Admin Panel",size=40)
    criminal_dropdown_name = Dropdown()
    test_Text=Text("if you just train your model,\n This will select test folder automatically.",text_align="center")
    complete_dialog = AlertDialog(
        modal=True,
        content=Text("Train complete"),
        actions=[
            TextButton("Ok", on_click=close_dlg),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )

    #Page overlay
    # page.overlay.append(
    #     Container(
    #     content=detect_lv,
    #     border_radius=border_radius.all(20),
    #     margin=margin.only(top=200,left=260,bottom=10),
    #     #bgcolor="#D4F4FF",
    #     width=690,
    #     height=440,
    #     )
    # )
    
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
                                    content=Column(
                                        controls = get_criminal_name()
                                    ),
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
                                    Image(src=f"https://cdn-icons-png.flaticon.com/512/2206/2206368.png", width=100, height=100),
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
                                    TextButton(content=Text("Exit", color=colors.BLACK, size=15), on_click= lambda _: page.window_destroy())
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
flet.app(target=admin_panel)