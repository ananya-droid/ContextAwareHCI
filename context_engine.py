# context_engine.py

def get_action(gesture, context):

    # -------------------------
    # PRESENTATION MODE
    # -------------------------
    if context == "PRESENTATION":

        if gesture == "PEACE":
            return "NEXT SLIDE"

        elif gesture == "POINT":
            return "SELECT"

        elif gesture == "THUMBS UP":
            return "CONTINUE"

        elif gesture == "FIST":
            return "CANCEL"

        elif gesture == "OPEN PALM":
            return "PAUSE"


    # -------------------------
    # MEDIA MODE
    # -------------------------
    elif context == "MEDIA":

        if gesture == "PEACE":
            return "NEXT TRACK"

        elif gesture == "POINT":
            return "SELECT"

        elif gesture == "THUMBS UP":
            return "PLAY"

        elif gesture == "FIST":
            return "STOP"

        elif gesture == "OPEN PALM":
            return "PLAY / PAUSE"


    # -------------------------
    # GENERAL MODE
    # -------------------------
    elif context == "GENERAL":

        if gesture == "PEACE":
            return "NEXT"

        elif gesture == "POINT":
            return "SELECT"

        elif gesture == "THUMBS UP":
            return "CONTINUE"

        elif gesture == "FIST":
            return "CANCEL"

        elif gesture == "OPEN PALM":
            return "PAUSE"


    return "NO ACTION"