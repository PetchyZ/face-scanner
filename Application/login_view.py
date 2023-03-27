from flet import Column, Ref, Text, ElevatedButton, Row, TextField
    
def login_click():
    print("login click",)
    user_name.current.value = ""
    password.current.value = ""
    page.update()
        
def register_click(e):
    page.clear()
        
        
def exit_click():
    print("In exit")
    #page.go("/exit")
            
def view_pop(view):
    page.views.pop()
    top_view = page.views[-1]
    page.go(top_view.route)
            
#page.bgcolor = "#D8FFFD"

#input
user_name = Ref[TextField]()
password = Ref[TextField]()
    
#result
info = Ref[Column]()

def login(page):
    login = [Row([Text(value="Welcome", style="headlineLarge")], alignment="center"),
    Row([TextField(ref=user_name, label="Username", width=400)],alignment="center"),
    Row([TextField(ref=password, label="Password", password=True, can_reveal_password=True, width=400)], alignment="center"),
    Row([ElevatedButton("Login", on_click=login_click(page)), ElevatedButton("Register", on_click=lambda _: page.go("/register")), ElevatedButton("Exit", on_click=exit_click(page))], alignment="center"),
    Column(ref=info),
    ]
    return login


