# This class implements database access to GitHub's REST-API for search queries

import requests
import json
import pandas as pd
import pymongo

if __name__ == "__main__":

    # Configure number and size of requests
    n_search_requests = 3
    n_results_per_page = 9

    # Initialize repository list
    repo_list = []

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    # Example search: https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:GilbertoEspinoza/emojify
    # search_terms = ['CNN', 'cnn', 'keras', 'Keras']
    search_terms = ['emojify']
    query_search_terms = '+'.join(search_terms)

    # search_locations = ['description', 'readme']
    search_locations = ['readme']
    query_search_locations = '+'.join(['in:' + location for location in search_locations])
    query = query_search_terms + '+' + query_search_locations  # + '&sort=stars&order=desc'

    # Retrieve local access token for GitHub API access
    with open('GitHub_Access_Token.txt', 'r') as f:
        access_token = f.read()

    print(access_token)


    # Specify request header for authentication
    headers = {'Authorization': 'token ' + access_token}

    print('Total number of requests: %d' % n_search_requests)
    print('Number of items per page: %d \n\n' % n_results_per_page)

    for page in range(0, n_search_requests):
        # Create url from query
        url = 'https://api.github.com/search/repositories?page=' + str(
            page) + '&per_page=' + str(n_results_per_page) + '&q=' + query

        # Submit request and save response
        response = requests.get(url, headers=headers)
        print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        print()

        # Print status code and encoding
        if response.status_code == 200:
            print('Request %d/%d successful' % (page + 1, n_search_requests))

            # Create json file from response
            json_data = json.loads(response.text)

            # Append repository to list
            repo_list.extend(json_data.get('items', {}))

        else:
            print('Request %d/%d failed. Error code: %d - %s' % (
            page + 1, n_search_requests, response.status_code, response.reason))

    # Save json file locally
    with open('Repo_Search_Results.json', 'w', encoding='utf8') as json_file:
        json.dump(repo_list, json_file)

    # Open file and display data
    # with open('Repo_Search_Results.json', 'r', encoding='utf8') as json_file:
    #     data = json.load(json_file)
    #
    # data_frame = pd.DataFrame(data)
    # print(data_frame.get('html_url', {}))

    # Initialize data dict for json export
    data = []

    # Create repository items and add to data list
    for repo in repo_list:
        query_url = 'https://api.github.com/search/code?q=extension:h5+repo:' + repo['full_name']
        print(repo['full_name'])
        response = requests.get(query_url, headers=headers)
        print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        print()
        has_architecture = False
        if response.status_code == 200:
            has_architecture = json.loads(response.text).get('total_count') > 0
            print('Checked for architecture. Result: %s \n' % str(has_architecture))
            print(response.text)
        else:
            print('Request for architecture failed. Error code: %d - %s' % (response.status_code, response.reason))
            print()

        if has_architecture:
            item = {'repo_url': repo['html_url'],
                    'repo_name': repo['name'],
                    'repo_owner': repo['owner']['login'],
                    'repo_desc': repo['description'],
                    'repo_ext_links': None,
                    'repo_last_mod': repo['updated_at'],
                    'repo_watch': repo['watchers_count'],
                    'repo_forks': repo['forks_count']}

            data.append(item)

    # Retrieve database credentials
    cred_path = 'C:/Users/svenk/PycharmProjects/GitHub_Scraping/connection_creds.txt'
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    # Establish database connection
    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.repos
    print(client.server_info())

    # Insert data into database if list is not empty
    if data:
        collection.insert_many(data)

    # Check database size and print to console
    print('Number of Repositories in Database: %d' % collection.count_documents({}))
