def recognize_gesture(landmarks):

    # Finger landmark indexes
    # Thumb: 4
    # Index: 8
    # Middle: 12
    # Ring: 16
    # Pinky: 20

    # MCP joints
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Finger tips
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check whether fingers are extended
    index_open = index_tip.y < index_mcp.y
    middle_open = middle_tip.y < middle_mcp.y
    ring_open = ring_tip.y < ring_mcp.y
    pinky_open = pinky_tip.y < pinky_mcp.y

    fingers = [
        index_open,
        middle_open,
        ring_open,
        pinky_open
    ]

    # Count extended fingers
    count = sum(fingers)

    # -------------------------
    # THUMBS UP
    # -------------------------
    # Thumb tip is above the thumb MCP
    # while the other four fingers are closed.

    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    thumb_up = thumb_tip.y < thumb_mcp.y

    if thumb_up and count == 0:
        return "THUMBS UP"

    # -------------------------
    # OPEN PALM
    # -------------------------
    if count == 4:
        return "OPEN PALM"

    # -------------------------
    # FIST
    # -------------------------
    if count == 0:
        return "FIST"

    # -------------------------
    # PEACE
    # -------------------------
    if index_open and middle_open and not ring_open and not pinky_open:
        return "PEACE"

    # -------------------------
    # POINT
    # -------------------------
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "POINT"

    # -------------------------
    # UNKNOWN
    # -------------------------
    return "UNKNOWN"
