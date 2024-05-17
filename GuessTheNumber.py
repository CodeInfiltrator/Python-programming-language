import random

welcome_message = "WELCOME TO CUPPY GAMES!"
cuppy_position = random.randint(1,5)

print("*****************************")
print(f"** {welcome_message} **")
print("*****************************")

while True:
    name = input("input your name: ")
    try:
        int(name)
        print("Name cannot be a number. Please enter a valid name.")
    except ValueError:
        if name.strip():
            break
        else:
            print("Name cannot be empty. Please enter a valid name.")

print(f'''
        hello {name}! Please looking at the cave below!!!
        |_| |_| |_| |_| |_|
        ''')

while True:
    user_option = input("What number cave do you think Cuppy is in? [1 / 2 / 3 / 4 / 5]: ")
    if user_option.isdigit() and 1 <= int(user_option) <= 5:
        break
    else:
        print("Invalid input. Please choose a number between 1 and 5")

while True:
    validation = input(f"are you sure you chose '{user_option}'? [y/n]: ")
    if (validation.lower() == "n"):
        user_option = input("Please choose again confidently: [1 / 2 / 3 / 4 / 5]: ")
    elif (validation.lower() == 'y'):
        break
    else:
        print("Please input correctly either y or n.")

print(f"You chose number: {user_option}")
if (int(user_option) == cuppy_position):
    print(f"Congrats {name}! you found Cuppy in cave number {cuppy_position}!")
else:
    print(f"You are wrong!, Cuppy is not there, Cuppy is in cave number: {cuppy_position} ")