import requests

url = 'http://127.0.0.1:8000/generate'
payload = {'inputs': 'Who are you?', "parameters":{"max_new_tokens": 4}}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
