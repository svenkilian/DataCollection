# This class implements database access to GitHub's REST-API for search queries

import requests
import json

if __name__ == "__main__":
    '''
    Main method to be invoked when file is executed
    '''

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    query = 'CNN+keras+in:readme&sort=stars&order=desc'

    # Create url from query
    url = 'https://api.github.com/search/repositories?q=' + query

    # Retrieve local access token for GitHub API access
    with open('GitHub_Access_Token.txt', 'r') as f:
        access_token = f.read()

    # Specify request header for authentication
    headers = '"Authorization: token e2db3762c08d2dcb1745478a2ed3aad254fde852" https://api.github.com'

    # Submit request and save response
    response = requests.get(url, headers)

    # Print status code and encoding
    print(response.status_code)
    print(response.encoding)

    # Create json file from response
    json_data = json.loads(response.text)
    # print(json_data)

    # Save json file locally
    with open('DataDump.json', 'w', encoding='utf8') as jason_file:
        json.dump(json_data, jason_file)
