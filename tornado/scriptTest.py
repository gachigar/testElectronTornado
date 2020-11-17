import os
import json

files = os.listdir(os.getcwd() + "/files/")
print(json.dumps(files))