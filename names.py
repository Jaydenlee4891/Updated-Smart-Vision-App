"""
names.py — Class index → label mapping for the custom 26-class model,
plus human-friendly TTS alert phrases.
"""

# ---------------------------------------------------------------------------
# Class map  (matches model output indices 0-25)
# ---------------------------------------------------------------------------
NAMES: dict[int, str] = {
    0:  "Bike Front",
    1:  "Bike Left",
    2:  "Bike Right",
    3:  "Car Front",
    4:  "Car Left",
    5:  "Car Right",
    6:  "Crossroad",
    7:  "Fence Front",
    8:  "Fence Left",
    9:  "Fence Right",
    10: "Pedestrian Light Green",
    11: "Pedestrian Light Red",
    12: "Person Front",
    13: "Person Left",
    14: "Person Right",
    15: "Pole Front",
    16: "Pole Left",
    17: "Pole Right",
    18: "Traffic Cone Right",
    19: "Traffic Light Green",
    20: "Traffic Light Orange",
    21: "Traffic Light Red",
    22: "Trash Left",
    23: "Trash Right",
    24: "Tree Front",
    25: "Tree Right",
}

# ---------------------------------------------------------------------------
# Spoken alert phrases (natural language, not raw class names)
# ---------------------------------------------------------------------------
ALERT_PHRASES: dict[str, str] = {
    "Bike Front":              "Bike ahead",
    "Bike Left":               "Bike on your left",
    "Bike Right":              "Bike on your right",
    "Car Front":               "Car ahead",
    "Car Left":                "Car on your left",
    "Car Right":               "Car on your right",
    "Crossroad":               "Crossroad ahead",
    "Fence Front":             "Fence ahead",
    "Fence Left":              "Fence on your left",
    "Fence Right":             "Fence on your right",
    "Pedestrian Light Green":  "Green light, safe to cross",
    "Pedestrian Light Red":    "Red light, stop",
    "Person Front":            "Person directly ahead",
    "Person Left":             "Person on your left",
    "Person Right":            "Person on your right",
    "Pole Front":              "Pole ahead",
    "Pole Left":               "Pole on your left",
    "Pole Right":              "Pole on your right",
    "Traffic Cone Right":      "Traffic cone on your right",
    "Traffic Light Green":     "Traffic light is green",
    "Traffic Light Orange":    "Traffic light is orange",
    "Traffic Light Red":       "Traffic light is red",
    "Trash Left":              "Trash bin on your left",
    "Trash Right":             "Trash bin on your right",
    "Tree Front":              "Tree ahead",
    "Tree Right":              "Tree on your right",
}
