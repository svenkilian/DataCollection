# Send the requests
import requests
import json

query = 'CNN+keras+in:readme&sort=stars&order=desc'
url = 'https://api.github.com/search/repositories?q=' + query
headers = '"Authorization: token e2db3762c08d2dcb1745478a2ed3aad254fde852" https://api.github.com'
response = requests.get(url, headers)

# Get the result
print(response.status_code)  # 200
print(response.encoding)  # 'Big5'


json_data = json.loads(response.text)
print(json_data)

with open('DataDump.json', 'w', encoding='utf8') as jason_file:
    json.dump(json_data, jason_file)
