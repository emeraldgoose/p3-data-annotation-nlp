import requests

tagtogAPIUrl = "https://www.tagtog.net/-api/documents/v1"

auth = requests.auth.HTTPBasicAuth(username="emeraldgoose", password="ultrasn0w")
params = {"owner": "emeraldgoose", "project": "data-annotation-nlp-10", "output": "null"}

# files = [("files", open('data-annotation-nlp-level3-nlp-10/files/item1.txt')), ("files", open('data-annotation-nlp-level3-nlp-10/files/item2.txt')), ("files", open('data-annotation-nlp-level3-nlp-10/files/item3.txt'))]

files = []
for i in range(243):
    files.append(("files", open(f'data-annotation-nlp-level3-nlp-10/files/item{i}.txt')))

response = requests.post(tagtogAPIUrl, params=params, auth=auth, files=files)
print(response.text)