import json
import os

# Import the app configurations using a path relative to this file's location
_base_dir = os.path.dirname(os.path.abspath(__file__))
configs = json.load(open(os.path.join(_base_dir, "..", "configs.json"), "r", encoding="utf-8"))