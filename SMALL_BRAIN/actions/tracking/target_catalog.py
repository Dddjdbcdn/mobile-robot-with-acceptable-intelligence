"""Trackable target catalogs and name normalization."""

HUMAN_RETARGETS = {
    "person": ["torso_center"],
    "human": ["torso_center"],
    "torso_center": ["torso_center"],
    "face": ["nose"],
    "head": ["nose"],
    "eye": ["left_eye", "right_eye"],
    "eyes": ["left_eye", "right_eye"],
    "left_eye": ["left_eye"],
    "right_eye": ["right_eye"],
    "hand": ["left_wrist", "right_wrist"],
    "left_hand": ["left_wrist"],
    "right_hand": ["right_wrist"],
    "leg": ["left_knee", "right_knee"],
    "left_leg": ["left_knee"],
    "right_leg": ["right_knee"],
    "feet": ["left_ankle", "right_ankle"],
    "left_feet": ["left_ankle"],
    "right_feet": ["right_ankle"],
}

HUMAN_TRACKABLE_PARTS = tuple(HUMAN_RETARGETS)

OBJECT_ALIASES = {
    "guitars": "guitar",
    "acoustic guitar": "guitar",
    "electric guitar": "guitar",
    "chairs": "chair",
    "seat": "chair",
    "seats": "chair",
    "tables": "table",
    "desk": "table",
    "desks": "table",
    "laptops": "laptop",
    "notebook": "laptop",
    "notebook computer": "laptop",
    "computer": "laptop",
    "doors": "door",
    "tv": "television",
    "t v": "television",
    "television set": "television",
    "screen": "television",
    "fans": "fan",
    "electric fan": "fan",
    "bottles": "bottle",
    "water bottle": "bottle",
    "water bottles": "bottle",
    "drink bottle": "bottle",
    "drinking bottle": "bottle",
    "mirrors": "mirror",
    "tool box": "toolbox",
    "tool boxes": "toolbox",
    "toolboxes": "toolbox",
    "dumbbells": "dumbbell",
    "weight": "dumbbell",
    "weights": "dumbbell",
    "hand weight": "dumbbell",
    "cameras": "camera",
    "webcam": "camera",
    "house plant": "houseplant",
    "house plants": "houseplant",
    "plant": "houseplant",
    "plants": "houseplant",
    "potted plant": "houseplant",
    "indoor plant": "houseplant",
    "curtains": "curtain",
    "drape": "curtain",
    "drapes": "curtain",
    "socket": "power socket",
    "sockets": "power socket",
    "power outlet": "power socket",
    "outlet": "power socket",
    "wall outlet": "power socket",
    "electrical outlet": "power socket",
    "plug socket": "power socket",
    "books": "book",
    "mic": "microphone",
    "mics": "microphone",
    "microphones": "microphone",
    "phone": "smartphone",
    "phones": "smartphone",
    "smart phone": "smartphone",
    "mobile phone": "smartphone",
    "cell phone": "smartphone",
    "cellphone": "smartphone",
    "air conditioning": "air conditioner",
    "aircon": "air conditioner",
    "air con": "air conditioner",
    "ac": "air conditioner",
    "a c": "air conditioner",
    "remote": "remote control",
    "remotes": "remote control",
    "controller": "remote control",
    "tv remote": "remote control",
    "television remote": "remote control",
    "remote controller": "remote control",
}


def normalize_human_target(target):
    normalized = str(target or "").lower().strip()
    normalized = normalized.replace("'s", "").replace("-", " ").replace("_", " ")
    words = normalized.split()

    side = None
    if "left" in words:
        side = "left"
    elif "right" in words:
        side = "right"

    if "torso" in words or "body" in words:
        part = "torso_center"
    elif "hand" in words or "wrist" in words:
        part = "hand"
    elif "leg" in words:
        part = "leg"
    elif "foot" in words or "feet" in words or "ankle" in words:
        part = "feet"
    elif "eye" in words or "eyes" in words:
        part = "eye"
    elif "face" in words:
        part = "face"
    elif "head" in words:
        part = "head"
    elif any(
        word in words
        for word in ("person", "human", "user", "me", "you", "dang", "owner")
    ):
        part = "person"
    else:
        return None

    if side is not None and part in {"hand", "leg", "eye", "foot"}:
        return f"{side}_{part}"
    return part


def normalize_object_target(target):
    normalized = str(target or "").lower().strip()
    normalized = normalized.replace("'s", "").replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return OBJECT_ALIASES.get(normalized, normalized)
