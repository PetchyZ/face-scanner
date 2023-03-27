import flet
import time
from . import login_view
from . import login_check
from .admin_page import admin_panel
from .user_page import user_panel
from flet import AlertDialog, TextButton, padding, border, Container, margin, alignment, Dropdown, dropdown, Column, ElevatedButton, Page, Text, View, colors, Row, TextField
import Database.database_manage as dm
panel_mode = ""

def main(page: Page):

    #super admin
    print("Start main")
    def super_admin_admin_panel_clicked(e):
        global panel_mode
        panel_mode = "admin"
        super_admin_modal.open = False
        page.update()
        page.window_destroy()
    
    def super_admin_user_panel_clicked(e):
        global panel_mode
        panel_mode = "user"
        super_admin_modal.open = False
        page.update()
        page.window_destroy()
    
    def super_admin_register_clicked(e):
        page.go("/register")
        page.update()
        super_admin_modal.open = False

    #dialog
    def close_dlg(e):
        dlg_modal.open = False
        page.update()
        
    def open_dlg_modal(e):
        page.dialog = dlg_modal
        dlg_modal.open = True
        page.update()

    super_admin_modal = AlertDialog(
        modal=True,
        title=Text("Confirm"),
        content=Text("Select your destination"),
        actions=[
            TextButton("Admin Panel", on_click= super_admin_admin_panel_clicked),
            TextButton("User Panel", on_click= super_admin_user_panel_clicked),
            TextButton("Register", on_click= super_admin_register_clicked)
        ],
        actions_alignment="end",
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )

    #login functions
    def login_clicked(e):
        global panel_mode
        login_check_status = login_check.check(username, password)
        if login_check_status == "super_admin":
            print(login_check_status)
            page.go('/register')
        elif login_check_status == "admin":
            print(login_check_status)
            role_status.value = f"Admin Login"
            panel_mode = "admin"
            page.window_destroy()
        elif login_check_status == "user":
            print(login_check_status)
            role_status.value = f"User Login"
            panel_mode = "user"
            page.window_destroy()
        page.update()
    
    #register functions
    def register_successful(e):
        close_dlg(e)
        #enc_password = dm.encrypting_password(password_reg.value)
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

    def route_change(route):
        if dm.add_first_user():
            print("Add first user")
            admin_user.value = "Username: {}".format(dm.get_super_admin()['username'])
            admin_password.value = "Password: {}".format(dm.get_super_admin()['password'])
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
                            Row([login_status], alignment="center"),
                            Row([admin_user], alignment="center"),
                            Row([admin_password], alignment="center"),
                            Row([role_status],alignment="center"),
                        Column(ref=login_view.info)]
                    ),
                    padding=padding.only(top=70),        
                    ),
                    alignment=alignment.center,
                    margin=margin.only(top=110,left=220),
                    bgcolor="#FFFFFF",
                    border=border.all(1,colors.BLACK), 
                    border_radius=30,
                    width=800,
                    height=400,
                    )
                    ],
                    bgcolor="#D8FFFD"
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
                        Container(
                            content=Column([
                                Row([Text(value="Register", style="headlineLarge", )], alignment="center"),
                                Row([username_reg], alignment="center"),
                                Row([password_reg], alignment="center"),
                                Row([c_password],alignment="center"),
                                Container(
                                    content=Row([role,],
                                    #alignment="center",
                                ),
                                margin=margin.only(left=200)
                        ),
                        Container(
                            content=Row([ElevatedButton("Back", on_click=lambda _: page.go("/")),ElevatedButton("Register", on_click=register_clicked),ElevatedButton("Admin panel", on_click=super_admin_admin_panel_clicked), ElevatedButton("User panel", on_click=super_admin_user_panel_clicked)], 
                        ),
                        margin=margin.only(left=200)
                        )
                        ],
                        
                        ),
                        padding=padding.only(top=40),    

                        ),
                        alignment=alignment.center,
                        margin=margin.only(top=90,left=240),
                        bgcolor="#FFFFFF",
                        border=border.all(1,colors.BLACK), 
                        border_radius=30,
                        width=800,
                        height=500,
                        )
                        
                    ],
                    bgcolor="#D8FFFD"
                    )
                )
            
            print("Register")
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    #Global variable
    #Global login variable   
    username = TextField(ref=login_view.user_name, label="Username", width=400)
    password = TextField(ref=login_view.password, label="Password", password=True, can_reveal_password=True, width=400)
    admin_user = Text("",selectable=True)
    admin_password = Text("",selectable=True)
    login_status = ElevatedButton("Login", on_click=login_clicked, data=False)
    role_status = Text()
     
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
                        ],
                    width=400
                    )                                              
    
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
    login_loading_dialog = AlertDialog(
        title=Column([
            Row([flet.ProgressRing(),Text("Login\nPlease wait...")],spacing=20),
        ],
        alignment='center'
        ),
        on_dismiss=lambda e: print("Modal dialog dismissed!"),
    )

    print("Now: on route chang")
    page.on_route_change = route_change
    print("Now: on view pop")
    page.on_view_pop = view_pop
    print("Now: route")
    page.go(page.route)

    #Global settings
    page.title = "Face Recognitioni Scanner"
    page.bgcolor = "#D8FFFD"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.window_center()
    page.update()

dm.get_ready()

def run():
    flet.app(target=main)
    if panel_mode == "admin":
        flet.app(target=admin_panel,assets_dir=r"./Application/assets")
    elif panel_mode == "user":
        flet.app(target=user_panel,assets_dir=r"./Application/assets")
    print("Exit Program")

# if __name__ == "__main__":
#     run()