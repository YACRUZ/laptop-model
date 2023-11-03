import requests
body = {
    "Ram":8,
    "Weight":1.86,
    "TouchScreen":0,
    "Ips":1,
    "Ppi":154.875632,
    "HDD":0,
    "SSD":256
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.22275930091861187}
