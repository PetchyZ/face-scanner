import path.path_settings as ps
#camera
import os
import Database.database_manage as dbm
import Model.face_processing as fp
import Model.model_manage as mm
from mtcnn import MTCNN
import tensorflow as tf
import numpy as np
from time import sleep
import cv2
import pickle
from PIL import Image
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
import joblib
import pymongo


import flet as ft
from flet import ListView, AlertDialog, TextButton, VerticalDivider, Divider, margin, border_radius, Image,padding,alignment,Container,Dropdown, dropdown, Column, Text, ElevatedButton,colors, Page, Row, icons
from time import sleep

frame_num = 0
def user_panel(page: Page): 
    
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
            p_protxt = ps.get_p_protxt()
            p_caffemodel = ps.get_p_caffemodel()

            #Mask model
            model = tf.keras.models.load_model(p_model)
            
            #facenet detec model
            faceNet = cv2.dnn.readNet(p_protxt,p_caffemodel)

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
                        myclient = pymongo.MongoClient(r"mongodb+srv://parit:132456@cluster1.gktrrcl.mongodb.net/?retryWrites=true&w=majority")
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

                            #append faced
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

    #Display function
    def add_criminal():
        criminal_name = dbm.list_criminal()
        criminal_dropdown_name.options = []
        #list image files
        image_files = []
        for i in range(1, len(criminal_name) + 1):
            if dbm.check_non_criminal_categories(criminal_name[i-1]):
                add_criminal_options(criminal_name[i-1])
                add_profile_dialog(criminal_name[i-1])
                criminal_box_lv = ft.ListView(expand=1, horizontal=True)
                criminal_box_lv.controls.append(
                    Container(
                                content=
                                    Row([
                                            
                                            Image(src=r"/criminal_images/{}.png".format(criminal_name[i-1]), width=100, height=90),
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
    
    def add_profile_dialog(criminal_name):
        profile_cri_info_dialog_lv = ft.ListView(expand=1, spacing=20, padding=padding.only(left=10,right=0,top=10,bottom=10), auto_scroll=True)
        profile_cri_contact_dialog_lv = ft.ListView(expand=1, spacing=20, padding=padding.only(left=10,right=0,top=10,bottom=10), auto_scroll=True)
        if dbm.check_non_criminal_categories(criminal_name):
            profile_data = dbm.list_criminal_profile(criminal_name)
            profile_cri_info_dialog_lv.controls.append(ft.Column([
                                            ft.Image(src=r'/criminal_images/{}.png'.format(criminal_name),width=150,height=150),
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
            
    #Update function
    def user_update(e):
        page.dialog = loading_update_dialog
        loading_update_dialog.open = True
        page.update()
        sleep(1)
        dbm.model_update()
        dbm.download_face_criminals()
        open_dlg_modal(e)
    
    #dialog
    def close_dlg(e):
        complete_dialog.open = False
        page.update()
     
    def open_dlg_modal(e):
        loading_update_dialog.open = False
        page.update()
        sleep(1)
        page.dialog = complete_dialog
        complete_dialog.open = True
        page.update()
    
    #camera detect function
    def camera_detect(e):
        print("Enter camera detect")
        page.dialog = loading_camera_dialog
        loading_camera_dialog.open = True
        page.update()  
        camera_on(True)

    ####  show right panel function  ####
    def camera_clicked(e):
        mode.value = f"Camera"
        right_panel.update()
    
    def update_clicked(e):
        user_update(e)
    
    def profile_clicked(e):
        for each_profile in profile_dialog:
            if each_profile['name'] == e.control.data:
                profile_detail = each_profile['profile_detail']
        print(profile_detail)
        page.dialog = profile_detail
        profile_detail.open = True
        page.update()

    #Global settings
    page.title = "Face Recognitioni Scanner"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.window_center()
    page.bgcolor = "#D8FFFD"
    
    #Global variable
    detect_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    criminal_lv = ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
    mode = Text(value="User Panel",size=40)
    criminal_dropdown_name = Dropdown()
    test_Text=Text("if you just train your model,\n This will select test folder automatically.",text_align="center")
    complete_dialog = AlertDialog(
        modal=True,
        title=Text("Update complete"),
        content=Text("Please restart your application.",size=20),
        actions=[
            TextButton("Ok", on_click=close_dlg),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    
        #dialog
    profile_dialog = []
    loading_camera_dialog = ft.AlertDialog(
        title=Column([
            Row([ft.ProgressRing(),Text("Loading Camera\nPlease wait...")],spacing=20),
        ],
        alignment='center'
        ),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    loading_camera_dialog.title_padding = padding.only(top=26,left=40)
    
    loading_update_dialog = ft.AlertDialog(
        title=Column([
            Row([ft.ProgressRing(),Text("Updating\nPlease wait...")],spacing=20),
        ],
        alignment='center'
        ),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    loading_update_dialog.title_padding = padding.only(top=26,left=40)
    
    page.overlay.append(
        Container(
        content=detect_lv,
        border_radius=border_radius.all(20),
        margin=margin.only(top=200,left=260,bottom=10),
        bgcolor="#D4F4FF",
        width=690,
        height=440,
        )
    )
    
    #right panel
    right_panel = Row([
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
                                    content=get_criminal(),
                                    # Column(
                                    #     controls = get_criminal()
                                    # ),
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
                                    Image(src="/{}.png".format("user"), width=100, height=100),
                                    Text(value="User",weight="w400",size=30),
                                    Divider(thickness=1, color="white")
                                    ],
                                    horizontal_alignment ="center"
                                    ),
                                padding=padding.only(top=10),
                                margin=margin.only(bottom=80),
                                alignment=alignment.center
                                ),
                        Container(content=
                                  Column([
                                    TextButton(content=Text("Camera", color=colors.BLACK, size=20),
                                        on_click=camera_clicked,
                                        ),
                                    TextButton(content=Text("Update", color=colors.BLACK, size=20),
                                        on_click=update_clicked,
                                        ),
                                    ],
                                    horizontal_alignment ="center"
                                    ),
                                margin=margin.only(bottom=110),
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
#ft.app(target=user_panel,assets_dir="assets")
