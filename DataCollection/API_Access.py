# This class implements database access to GitHub's REST-API for search queries
from config import ROOT_DIR
import os
import datetime
import json
import time
from math import ceil, floor
import requests
from Helper_Functions import print_progress
import DataCollection


def iterate_pages(resp):
    """
    Iterates over all pages following the 'next' links of ab API response
    :param resp: response of previous request
    :return:
    """
    global token_counter, rotation_cycle, page_counter
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
        print_progress(page_counter, ceil(n_results / 100.0), prog='Last request: %g' % round(time_diff, 2),
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
    # Specify words that are excluded from search
    stop_words = ['tutorial']
    query_stop_words = 'NOT+' + '+NOT+'.join(stop_words)

    search_locations = ['title', 'readme', 'description']  # Specify locations to search
    query_search_locations = '+'.join(['in:' + location for location in search_locations])
    search_from_date = datetime.date(2019, 5, 18)  # Keras release date: 2015-03-27
    query_search_from = 'created:>=' + search_from_date.isoformat()
    query_sort_by = 'score'  # updated, stars, forks, default: score
    query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
            query_search_from + '&sort=' + query_sort_by + '&order=desc'

    # Retrieve local access token for GitHub API access
    with open('DataCollection/credentials/GitHub_Access_Token.txt', 'r') as f:
        access_tokens = f.readlines()

    # List tokens
    access_tokens = [token.rstrip('\n') for token in access_tokens]

    # Determine length of rotation cycle
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
    date_tomorrow = datetime.date.today() + datetime.timedelta(days=1)  # Search end data (= tomorrow's date)
    date_today = datetime.date.today()
    time_delta = (date_tomorrow - search_from_date).days  # Search time period length in days

    avg_daily_results = n_results / time_delta  # Average number of search results per day
    avg_period = 1000 / avg_daily_results  # Average period length yielding 1000 search results (API limit)

    print('\nAverage daily results: ' + str(avg_daily_results))
    print('Average period: ' + str(avg_period))

    # Set length for search period to average period yielding 1000 results
    period_length = datetime.timedelta(days=avg_period)
    print('Period length (days): %d\n\n' % period_length.days)

    # Specify last date for current search iteration
    last_date = search_from_date + period_length

    # If last date is in the future, set last_data to date_tomorrow
    if last_date > date_today:
        last_date = date_today

    # Iterate over time deltas
    # Make URL request to GitHub API by date range chunks (due to limit of 1000 results per search request)
    while True:
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
        # print('\nNumber of Repos: %d' % repo_count)

        # If number of found repositories exceeds limit of 1000 search results, reduce period_length
        slack = 0.8
        if repo_count > 1000:
            delta_period_length = datetime.timedelta(
                days=floor(((repo_count / 1000.0) - slack) * period_length.days))
            last_date = last_date - delta_period_length
            period_length -= delta_period_length
            print('Change of period length to: %d' % period_length.days)
            continue
        else:
            pass

        # Create json file from response
        json_data = json.loads(response.text)

        # Append repositories to list
        repo_list.extend(json_data.get('items', {}))

        # Increase page counter
        page_counter += 1
        end_time = time.time()

        # Print search progress as progress bar
        print_progress(page_counter, ceil(n_results / 100.0),
                       prog='Last request (first of new): %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)

        # print('\n\nTime for request: %g' % (end_time - start_time))
        # print('Limit: %d' % int(response.headers['X-RateLimit-Limit']))
        # print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        # print('Token: %d,\n%s' % (token_index, access_tokens[token_index]))

        # Print status code and encoding
        if response.status_code == 200:
            # Iterate through returned pages
            iterate_pages(response)

        else:
            # If request unsuccessful, print error message
            print('Request failed. Error code: %d - %s' % (response.status_code, response.reason))

        # Set search date window to next time frame
        search_from_date = last_date + datetime.timedelta(days=1)

        # Check if end of search time frame is reached
        if last_date == date_today:
            break  # End search iterations
        elif last_date + period_length > date_today:
            last_date = date_today  # Restrict search time frame to past
        else:
            last_date += period_length  # If end not reached, increase last date by period length

    # Save json file locally in specified location
    # with open('Repo_Search_Results.json', 'w', encoding='utf8') as json_file:
    #     json.dump(repo_list, json_file)

    # Print number of saved repositories
    print('\n\nNumber of saved repositories: %d' % len(repo_list))

    # Initialize data list for json export
    data = []

    # Create repository items and add to data list
    true_count = 0
    false_count = 0

    repo_count = len(repo_list)

    # Iterate through repositories in list and save metadata
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
                'repo_full_name': repo['full_name'],
                'repo_owner': repo['owner']['login'],
                'repo_desc': repo['description'],
                'repo_ext_links': None,
                'repo_last_mod': repo['updated_at'],
                'repo_watch': repo['watchers_count'],
                'repo_forks': repo['forks_count'],
                'private': repo['private'],
                'repo_created_at': repo['created_at'],
                'homepage': repo['homepage'],
                'size': repo['size'],
                'language': repo['language'],
                'has_wiki': repo['has_wiki'],
                'license': repo['license'],
                'open_issues_count': repo['open_issues_count'],
                # 'subscribers_count': repo['subscribers_count'],
                'github_id': repo['id'],
                'is_fork': repo['fork']}

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

    # Create MongoDB collection instance
    collection = DataCollection.DataCollection('Repositories')
    # print(collection)

    # Insert data into database if list is not empty
    if data:
        collection.collection_object.insert_many(data)

    # Check database size and print to console
    print('Number of Repositories in Database: %d' % collection.collection_object.count_documents({}))
