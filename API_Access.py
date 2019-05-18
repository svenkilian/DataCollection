# This class implements database access to GitHub's REST-API for search queries
import datetime
import json
import sys
import time
from math import ceil, floor

import dateutil.parser
import pandas as pd
import pymongo
import requests

from Helper_Functions import print_progress


def iterate_pages(resp):
    global token_counter, rotation_cycle, page_counter, slack
    while 'next' in resp.links.keys():
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1
        headers = {'Authorization': 'token ' + access_tokens[token_index]}
        begin_query = time.time()
        # Submit request and save response
        resp = requests.get(resp.links['next']['url'], headers=headers)
        page_counter += 1
        end_time = time.time()
        time_diff = end_time - begin_query

        # Print search progress as progress bar
        print_progress(page_counter, ceil(n_results / (100 * slack)), prog='Last request: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)

        # Print status code and encoding
        if resp.status_code == 200:
            # If request successful

            # Create json file from response
            json_data = json.loads(resp.text)

            # Append repositories to list
            repo_list.extend(json_data.get('items', {}))

        else:
            # If request unsuccessful, print error message
            print('Request %d/%d failed. Error code: %d - %s' % (
                token_counter + 1, n_search_requests, resp.status_code, resp.reason))

    # print('\n\nLast page reached')


if __name__ == "__main__":

    # Configure number and size of requests (responses are paginated)
    n_search_requests = 10
    n_results_per_page = 100

    # Initialize repository list
    repo_list = []

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    # Example search: https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:GilbertoEspinoza/emojify
    # search_terms = ['CNN', 'cnn', 'keras', 'Keras', '"image+processing"', '"character+recognition"', 'forecasting']
    search_terms = ['keras']
    query_search_terms = '+'.join(search_terms)
    stop_words = ['tutorial']
    query_stop_words = 'NOT+' + '+NOT+'.join(stop_words)
    # query_stop_words = ''

    search_locations = ['title', 'readme', 'description']
    query_search_locations = '+'.join(['in:' + location for location in search_locations])
    search_from_date = datetime.date(2019, 4, 27)  # Keras release date: 2015-03-27
    query_search_from = 'created:>=' + search_from_date.isoformat()
    query_sort_by = 'score'  # updated, stars, forks, default: score
    query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
            query_search_from + '&sort=' + query_sort_by + '&order=desc'

    # Retrieve local access token for GitHub API access
    with open('GitHub_Access_Token.txt', 'r') as f:
        access_tokens = f.readlines()

    access_tokens = [token.rstrip('\n') for token in access_tokens]
    rotation_cycle = len(access_tokens)

    # print('Total number of requests: %d' % n_search_requests)
    # print('Number of items per page: %d \n\n' % n_results_per_page)

    # Create initial url from query
    url = 'https://api.github.com/search/repositories?page=1&per_page=' + str(n_results_per_page) + '&q=' + query

    # print(url)

    # Set results number and token counter to zero
    n_results = 0
    page_counter = 0
    token_counter = 0

    # Specify header with authentication for first query
    headers = {'Authorization': 'token ' + access_tokens[0]}

    # Increase token counter
    token_counter += 1

    # Start timer for search request and submit API request
    begin_query = time.time()
    start_time = time.time()
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # If request successful
        n_results = json.loads(response.text).get('total_count')  # Query total count from response
        print('\n\nTotal number of repositories found: %d\n' % n_results)
        print('Initial request successful')
    end_time = time.time()  # Stop timer
    time_diff = end_time - begin_query  # Calculate duration of query response

    # Calculate total time period to search
    date_tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    time_delta = (date_tomorrow - search_from_date).days

    avg_daily_results = n_results / time_delta
    avg_period = 1000 / avg_daily_results

    # Print out search progress as progress bar
    # print_progress(token_counter + 1, n_results / avg_period, prog='Request avg: %g' % round(time_diff, 2),
    #                time_lapsed=end_time - start_time)

    print('\nAverage daily results: ' + str(avg_daily_results))
    print('Average period: ' + str(avg_period))

    # Set length for search period chunks to 50 days
    period_length = datetime.timedelta(days=avg_period)
    print('Period length (days): %d\n\n' % period_length.days)
    last_date = search_from_date + period_length
    if last_date > date_tomorrow:
        last_date = date_tomorrow - datetime.timedelta(days=1)

    slack = avg_period / float(period_length.days)

    # Iterate over time deltas
    # Make URL request to GitHub API by date range chunks (due to limit of 1000 results per search request)
    while last_date < date_tomorrow:
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1  # Increase counter
        headers = {'Authorization': 'token ' + access_tokens[token_index]}

        time_frame = 'created:' + search_from_date.isoformat() + '..' + last_date.isoformat()
        # print('Timeframe: %s' % time_frame)
        # Create search query
        query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
                time_frame + '&sort=' + query_sort_by + '&order=desc'

        # Create initial url from query
        url = 'https://api.github.com/search/repositories?page=1&per_page=' + str(n_results_per_page) + '&q=' + query
        # Submit request and time response
        begin_query = time.time()
        response = requests.get(url, headers=headers)
        repo_count = json.loads(response.text).get('total_count')
        print('Number of Repos: %d' % repo_count)
        if repo_count > 1000:
            delta_period_length = datetime.timedelta(
                floor((1000 / float(repo_count)) * period_length.days)) - period_length
            last_date = last_date + delta_period_length
            period_length += delta_period_length
            print('Change of period length to: %d' % period_length.days)
            continue
        else:
            pass

        # Create json file from response
        json_data = json.loads(response.text)

        # Append repositories to list
        repo_list.extend(json_data.get('items', {}))

        page_counter += 1
        end_time = time.time()

        # Print search progress as progress bar
        print_progress(page_counter, ceil(n_results / (100 * slack)),
                       prog='Last request (first of new): %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)

        # print('\n\nTime for request: %g' % (end_time - start_time))
        # print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        # print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        # print('Token: %d,\n%s' % (token_index, access_tokens[token_index]))

        # Print status code and encoding
        if response.status_code == 200:
            iterate_pages(response)

        else:
            # If request unsuccessful, print error message
            print('Request failed. Error code: %d - %s' % (response.status_code, response.reason))

        search_from_date = last_date
        last_date += period_length

    # Save json file locally in specified location
    with open('Repo_Search_Results.json', 'w', encoding='utf8') as json_file:
        json.dump(repo_list, json_file)

    print('Number of saved repositories: %d' % len(repo_list))

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

    repo_count = len(repo_list)

    # Iterate through repositories in list
    for repo_index, repo in enumerate(repo_list):
        # Seach for 'save' in .py file
        # query_url = 'https://api.github.com/search/code?q=save+extension:py+repo:' + repo['full_name']

        token_index = token_counter % rotation_cycle
        token_counter += 1
        headers = {'Authorization': 'token ' + access_tokens[token_index]}

        # # Specify search for file extensions '.h5' and 'hdf5'
        # query_url = 'https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:' + repo['full_name']
        # print(repo['full_name'])
        # print('Repo %d/%d' % (repo_index + 1, repo_count))
        # start_time = time.time()
        # response = requests.get(query_url, headers=headers)
        # end_time = time.time()
        # print('Time for request: %g' % (end_time - start_time))
        #
        # has_architecture = False
        #
        # # Check for response
        # if response.status_code == 200:
        #     # If request to API successful
        #     has_architecture = json.loads(response.text).get('total_count') > 0  # '.h5' or '.hdf5' file found in repo
        #     print('Checked for architecture. Result: %s \n' % str(has_architecture))
        #
        #     # Print out remaining capacity
        #     print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        #     print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        #
        #     # Update success counters
        #     true_count = true_count + 1 if has_architecture else true_count
        #     false_count = false_count + 1 if not has_architecture else false_count
        #
        # else:
        #     # If request to API unsuccessful
        #     print('Request for architecture failed. Error code: %d - %s' % (response.status_code, response.reason))
        #     print()

        # Speicify repo meta data to be extracted from API response
        item = {'repo_url': repo['html_url'],
                'repo_name': repo['name'],
                'repo_owner': repo['owner']['login'],
                'repo_desc': repo['description'],
                'repo_ext_links': None,
                'repo_last_mod': repo['updated_at'],
                'repo_watch': repo['watchers_count'],
                'repo_forks': repo['forks_count']}

        # # Add architecture attribute
        # if has_architecture:
        #     item['has_structure'] = True
        #     # print(json.loads(response.text).get('items')[0])
        # else:
        #     item['has_structure'] = False

        # Append item to data list
        data.append(item)

    # # Print search success information
    # print('Search for architecture completed. \n'
    #       'True: %d \n' % true_count +
    #       'False: %d \n' % false_count)

    # Retrieve database credentials
    cred_path = 'connection_creds.txt'
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    # Establish database connection
    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.Repositories
    print(client.server_info())

    # Insert data into database if list is not empty
    if data:
        collection.insert_many(data)

    # Check database size and print to console
    print('Number of Repositories in Database: %d' % collection.count_documents({}))
