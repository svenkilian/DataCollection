# This class implements database access to GitHub's REST-API for search queries

import requests
import json
import pandas as pd
import pymongo

if __name__ == "__main__":

    # Configure number and size of requests (responses are paginated)
    n_search_requests = 1
    n_results_per_page = 29

    # Initialize repository list
    repo_list = []

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    # Example search: https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:GilbertoEspinoza/emojify
    # search_terms = ['CNN', 'cnn', 'keras', 'Keras']
    search_terms = ['keras', 'neural network', 'ann', 'artificial neural network']
    query_search_terms = '+'.join(search_terms)

    search_locations = ['readme', 'description']
    query_search_locations = '+'.join(['in:' + location for location in search_locations])
    query = query_search_terms + '+' + query_search_locations + '+language:python&sort=updated&order=desc'

    # Retrieve local access token for GitHub API access
    with open('GitHub_Access_Token.txt', 'r') as f:
        access_token = f.read()

    print(access_token)

    # Specify request header for authentication
    headers = {'Authorization': 'token ' + access_token}

    print('Total number of requests: %d' % n_search_requests)
    print('Number of items per page: %d \n\n' % n_results_per_page)

    # Make URL request to GitHub API by page (due to pagination of responses)
    for page in range(1, n_search_requests + 1):
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
            # If request successful
            print('Request %d/%d successful' % (page + 1, n_search_requests))

            # Create json file from response
            json_data = json.loads(response.text)

            # Append repositories to list
            repo_list.extend(json_data.get('items', {}))

        else:
            # If request unsuccessful, print error message
            print('Request %d/%d failed. Error code: %d - %s' % (
                page + 1, n_search_requests, response.status_code, response.reason))

    # Save json file locally in specified location
    with open('Repo_Search_Results.json', 'w', encoding='utf8') as json_file:
        json.dump(repo_list, json_file)

    # Open file and display data
    # with open('Repo_Search_Results.json', 'r', encoding='utf8') as json_file:
    #     data = json.load(json_file)
    #
    # data_frame = pd.DataFrame(data)
    # print(data_frame.get('html_url', {}))

    # Initialize data list for json export
    data = []

    # Create repository items and add to data list
    true_count = 0
    false_count = 0

    # Iterate through repositories in list
    for repo in repo_list:
        # Seach for 'save' in .py file
        # query_url = 'https://api.github.com/search/code?q=save+extension:py+repo:' + repo['full_name']

        # Specify search for file extensions '.h5' and 'hdf5'
        query_url = 'https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:' + repo['full_name']
        print(repo['full_name'])
        response = requests.get(query_url, headers=headers)
        print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))

        has_architecture = False

        # Check for response
        if response.status_code == 200:
            # If request to API successful
            has_architecture = json.loads(response.text).get('total_count') > 0  # '.h5' or '.hdf5' file found in repo
            print('Checked for architecture. Result: %s \n' % str(has_architecture))

            # Update success counters
            true_count = true_count + 1 if has_architecture else true_count
            false_count = false_count + 1 if not has_architecture else false_count

        else:
            # If request to API unsuccessful
            print('Request for architecture failed. Error code: %d - %s' % (response.status_code, response.reason))
            print()

        # Speicify repo meta data to be extracted from API response
        item = {'repo_url': repo['html_url'],
                'repo_name': repo['name'],
                'repo_owner': repo['owner']['login'],
                'repo_desc': repo['description'],
                'repo_ext_links': None,
                'repo_last_mod': repo['updated_at'],
                'repo_watch': repo['watchers_count'],
                'repo_forks': repo['forks_count']}

        # Add architecture attribute
        if has_architecture:
            item['has_structure'] = True
            # print(json.loads(response.text).get('items')[0])
        else:
            item['has_structure'] = False

        # Append item to data list
        data.append(item)

    # Print search success information
    print('Search for architecture completed. \n'
          'True: %d \n' % true_count +
          'False: %d \n' % false_count)

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
