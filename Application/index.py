import sys
#sys.path.insert(0, '/Face_recognition_Project/Database')
from importlib.machinery import SourceFileLoader
import flet
#from Database import database_manage as dm
import pymongo
import login_view
import login_check
from admin_page_test import admin_panel
from user_page import user_panel
from flet import AlertDialog, TextButton, padding, border, Container, margin, alignment, Dropdown, dropdown, Column, AppBar, ElevatedButton, Page, Text, View, colors, Row, TextField
import Database.database_manage as dm
panel_mode = ""

def main(page: Page):
    
    #Global settings
    page.title = "Face Recognitioni Scanner"
    page.bgcolor = "#D8FFFD"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.window_center()
    page.update()
    
    #dialog
    def close_dlg(e):
        dlg_modal.open = False
        page.update()
        
    def open_dlg_modal(e):
        page.dialog = dlg_modal
        dlg_modal.open = True
        page.update()
    
    #Global variable
    dlg_modal = AlertDialog(
        modal=True,
        title=Text("Register fail"),
        content=Text("Your password doesn't match!"),
        actions=[
            TextButton("Ok", on_click=close_dlg),
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )
    
    #login functions
    def login_clicked(e):
        global panel_mode
        login_check_status = login_check.check(username, password)
        if login_check_status == "admin":
            print(login_check_status)
            t.value = f"Admin Login"
            panel_mode = "admin"
            page.window_destroy()
            
        elif login_check_status == "user":
            print(login_check_status)
            t.value = f"User Login"
            panel_mode = "user"
            page.window_destroy()
        page.update()
    
    #register functions
    def register_successful(e):
        close_dlg(e)
        dm.send_user_data(username_reg,password_reg,role)
        page.go("/")
        
    def check_user(username):
        return dm.check_user_data(username)
        
    #register
    def register_clicked(e):
        if(password_reg.value == c_password.value):                                                                                                   
            if((username_reg.value != "") and (password_reg.value != "") ) : 
                if(role.value != None): 
                    if(check_user(username_reg.value) == False):
                        print("username_reg: ", username_reg.value)
                        print("password_reg: ", password_reg.value)
                        dlg_modal.title = Text("Register succesfully!")
                        dlg_modal.content = Text("we will send you to login page.")
                        dlg_modal.actions=[TextButton("Ok", on_click=register_successful)]
                        open_dlg_modal(e)
                    else:
                        dlg_modal.title = Text("Register fail")
                        dlg_modal.content = Text("User already exist.")
                        open_dlg_modal(e)
                else:
                    dlg_modal.content = Text("Select your role.")
                    open_dlg_modal(e)
            else:
                dlg_modal.content = Text("your username or password is empty.")
                open_dlg_modal(e) 
        else:
            open_dlg_modal(e)

        
    #Global login variable   
    username = TextField(ref=login_view.user_name, label="Username", width=400)
    password = TextField(ref=login_view.password, label="Password", password=True, can_reveal_password=True, width=400)
    login_status = ElevatedButton("Login", on_click=login_clicked, data=False)
    t = Text()
     
    #Global register variable
    username_reg = TextField(ref=login_view.user_name, label="Username", width=400)
    password_reg = TextField(ref=login_view.password, label="Password", password=True, can_reveal_password=True, width=400)
    c_password = TextField(label="Confirm Password", password=True, can_reveal_password=True, width=400)
    role = Dropdown(
                    label = "Roles",
                    hint_text = "Choose user role",
                    options = [
                        dropdown.Option("admin"),
                        dropdown.Option("user")
                        ]
                    )                                              
    
    def route_change(route):
        page.views.clear()
        page.views.append(
            View(
                "/",
                    [
                    Container(content=
                        Container(content=
                            Column([Row([Text(value="Welcome", style="headlineLarge")], alignment="center"),
                            Row([username],alignment="center"),
                            Row([password], alignment="center"),
                            Row([login_status, ElevatedButton("Register", on_click=lambda _: page.go("/register")),], alignment="center"),
                            Row([t],alignment="center"),
                        Column(ref=login_view.info)]
                    ),
                    padding=padding.only(top=70),        
                    ),
                    alignment=alignment.center,
                    margin=margin.only(top=110,left=220),
                    #bgcolor="#D8FFFD",
                    border=border.all(1,colors.BLACK), 
                    border_radius=30,
                    width=800,
                    height=400,
                    )
                    ]
            )
        )
        if page.route == "/register":
            username_reg.value = f""
            password_reg.value = f""
            c_password.value = f""
            role.value = f""
            page.views.append(
                View(
                    "/register",
                    [
                    Container(content=
                        Container(content=Column([
                        Row([Text(value="Register", style="headlineLarge", )], alignment="center"),
                        Row([username_reg], alignment="center"),
                        Row([password_reg], alignment="center"),
                        Row([c_password],alignment="center"),
                        Container(content=
                            Row([role,],
                            #alignment="center",
                        ),
                        margin=margin.only(left=200)
                        ),
                        Container(content=
                        Row([ElevatedButton("Back", on_click=lambda _: page.go("/")),ElevatedButton("Register", on_click=register_clicked)], 
                            #alignment="center",
                        ),
                        margin=margin.only(left=200)
                        )
                        ]),
                        padding=padding.only(top=40),         
                        ),
                        alignment=alignment.center,
                        margin=margin.only(top=90,left=240),
                        #bgcolor="#D8FFFD",
                        border=border.all(1,colors.BLACK), 
                        border_radius=30,
                        width=800,
                        height=500,
                        )
                    ]
                    )
                )
            
            print("Pass")
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    print("Now: on route chang")
    page.on_route_change = route_change
    print("Now: on view pop")
    page.on_view_pop = view_pop
    print("Now: route")
    page.go(page.route)

flet.app(target=main)
if panel_mode == "admin":
    flet.app(target=admin_panel,assets_dir="assets")
elif panel_mode == "user":
    flet.app(target=user_panel,assets_dir="assets")
print("Exit Program")