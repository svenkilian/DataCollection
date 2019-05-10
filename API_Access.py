# This class implements database access to GitHub's REST-API for search queries

import requests
import json
import pandas as pd

if __name__ == "__main__":

    # Configure number and size of requests
    n_search_requests = 10
    n_results_per_page = 100

    # Initialize repository list
    repo_list = []

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    # query = 'CNN+keras+in:readme&sort=stars&order=desc'
    query = 'extension:h5&sort=stars&order=desc'

    # Retrieve local access token for GitHub API access
    with open('GitHub_Access_Token.txt', 'r') as f:
        access_token = f.read()

    # Specify request header for authentication
    headers = '"Authorization: token ' + access_token + '"'

    print('Total number of requests: %d' % n_search_requests)
    print('Number of items per page: %d \n\n' % n_results_per_page)

    for page in range(0, n_search_requests):
        # Create url from query
        url = 'https://api.github.com/search/repositories?page=' + str(
            page) + '&per_page=' + str(n_results_per_page) + '&q=' + query

        # Submit request and save response
        response = requests.get(url, headers)

        # Print status code and encoding
        if response.status_code == 200:
            print('Request %d/%d succesful' % (page + 1, n_search_requests))

        # Create json file from response
        json_data = json.loads(response.text)

        # Append repository to list
        repo_list.extend(json_data.get('items', {}))

    # Save json file locally
    with open('Repo_Search_Results.json', 'w', encoding='utf8') as jason_file:
        json.dump(repo_list, jason_file)

    with open('Repo_Search_Results.json', 'r', encoding='utf8') as jason_file:
        data = json.load(jason_file)

    data_frame = pd.DataFrame(data)

    print(data_frame)
