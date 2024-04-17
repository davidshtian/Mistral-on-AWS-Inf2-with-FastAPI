import json
import requests
import torch
import numpy as np

image_embeds_saved = np.load('input_embeds.npy')
payload = {'inputs': image_embeds_saved.tolist()}

url = 'http://127.0.0.1:8000/generate'
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
