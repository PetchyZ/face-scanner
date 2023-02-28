import flet
from flet import alignment,Container,Dropdown, dropdown, Column, Ref, Text, theme,ElevatedButton,colors,IconButton, Page, Row, TextField, icons

def register(page: Page): 
    
    def register_click(e):
        page.add(Text(f"Register successfully"))
        
    #Global settings
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    
    #input
    user_name = Ref[TextField]()
    password = Ref[TextField]()
    c_password = Ref[TextField]()
    roles = Ref[Dropdown]()
    page.add(
        Row([Text(value="Register", style="headlineLarge")], alignment="center"),
        TextField(ref=user_name, label="Username", width=400),
        TextField(ref=password, label="Password", password=True, can_reveal_password=True, width=400),
        TextField(ref=c_password, label="Confirm Password", password=True, can_reveal_password=True, width=400),
        Row([
                Dropdown(ref=roles,
                label = "Roles",
                hint_text = "Choose user role",
                options = [
                    dropdown.Option("Admin"),
                    dropdown.Option("User")
                ],
                width=200),
                ElevatedButton("Register", on_click=register_click),
                ],alignment="center",
                spacing = 110
            ),
        )
flet.app(target=register)