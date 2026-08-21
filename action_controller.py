import subprocess
import time


def execute_action(action):

    if action == "NEXT":
        print("ACTION: NEXT")
        # Test action: open calculator
        subprocess.Popen("calc.exe")

    elif action == "CANCEL":
        print("ACTION: CANCEL")

    elif action == "CONTINUE":
        print("ACTION: CONTINUE")

    elif action == "SELECT":
        print("ACTION: SELECT")

    elif action == "STOP":
        print("ACTION: STOP")

    else:
        print("ACTION: NO ACTION")