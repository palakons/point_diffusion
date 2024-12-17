#write a script to call a url and print the response

import requests
import json

url = 'https://api.wandb.ai/files/palakons/point_cloud_diffusion/2j1z1z1j'
response = requests.get(url)
print(response.text)
