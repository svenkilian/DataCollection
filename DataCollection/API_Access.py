# This class implements database access to GitHub's REST-API for search queries
import re
import threading

from config import ROOT_DIR
import os
import datetime
import json
import time
from math import ceil, floor
import requests
from Helper_Functions import print_progress, split_time_interval, identify_language, get_access_tokens
from Readme_Scraper import get_readme
import DataCollection
from multiprocessing import Process


def iterate_pages(resp, tokens):
    """
    Iterates over all pages following the 'next' links of ab API response
    :param resp: response of previous request
    """
    global token_counter, rotation_cycle, page_counter, n_results, start_time, repo_list
    while 'next' in resp.links.keys():
        # JOB: Retrieve and process current results page
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1
        headers = {'Authorization': 'token ' + tokens[token_index]}
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
                token_counter + 1, n_results, resp.status_code, resp.reason))


def seach_repos(start_date, end_date, tokens):
    global token_counter, rotation_cycle, page_counter, n_results, start_time, repo_list
    # JOB: Specify search terms, time period to search, resources to search
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
    # stop_words = ['tutorial']
    # query_stop_words = 'NOT+' + '+NOT+'.join(stop_words)

    search_locations = ['title', 'readme', 'description']  # Specify locations to search
    query_search_locations = '+'.join(['in:' + location for location in search_locations])
    # search_from_date = datetime.date(2019, 5, 22)  # Keras release date: 2015-03-27
    search_from_date = start_date
    search_end_date = end_date
    query_search_from = 'created:>=' + search_from_date.isoformat()
    query_sort_by = 'score'  # updated, stars, forks, default: score
    # query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
    #         query_search_from + '&sort=' + query_sort_by + '&order=desc'

    time_frame = 'created:' + search_from_date.isoformat() + '..' + search_end_date.isoformat()
    query = query_search_terms + '+' + query_search_locations + '+language:python+' + \
            time_frame + '&sort=' + query_sort_by + '&order=desc'

    # JOB: Get authentication tokens from file
    # Retrieve local access token for GitHub API access
    # with open(os.path.join(ROOT_DIR, 'DataCollection/credentials/GitHub_Access_Token.txt'), 'r') as f:
    #     access_tokens = f.readlines()
    # # List tokens
    # access_tokens = [token.rstrip('\n') for token in access_tokens]
    # Determine length of rotation cycle
    rotation_cycle = len(tokens)

    # print('Total number of requests: %d' % n_search_requests)
    # print('Number of items per page: %d \n\n' % n_results_per_page)

    # JOB: Create initial query and send request
    # Create initial url from query
    url = 'https://api.github.com/search/repositories?page=1&per_page=' + str(n_results_per_page) + '&q=' + query
    # print(url)

    # Set results number and token counter to zero
    n_results = 0
    page_counter = 0
    token_counter = 0

    # Specify header with authentication for first query
    headers = {'Authorization': 'token ' + tokens[0]}

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
    date_today = datetime.date.today()  # Today's date
    # time_delta = (date_tomorrow - search_from_date).days  # Search time period length in days
    time_delta = (search_end_date - search_from_date).days + 1  # Search time period length in days

    # Calculate search period to use for iteration of results
    avg_daily_results = n_results / time_delta  # Average number of search results per day

    avg_period = 1000 / avg_daily_results  # Average period length yielding 1000 search results (API limit)

    # print('\nAverage daily results: ' + str(avg_daily_results))
    # print('Average period: ' + str(avg_period))

    # Set length for search period to average period yielding 1000 results
    period_length = datetime.timedelta(days=avg_period)
    # print('Period length (days): %d\n\n' % period_length.days)

    # Specify last date for current search iteration
    last_date = search_from_date + period_length

    # If last date is in the future, set last_data to date_tomorrow
    if last_date > search_end_date:
        last_date = search_end_date

    # JOB: Iterate over time periods and make API search requests
    # Make URL request to GitHub API by date range chunks (due to limit of 1000 results per search request)
    while True:
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1  # Increase counter
        headers = {'Authorization': 'token ' + tokens[token_index]}  # Specify header for authentication

        # Specify time frame to search
        time_frame = 'created:' + search_from_date.isoformat() + '..' + last_date.isoformat()
        # print('Time frame to search: %s' % time_frame)

        # Create search query with stop words
        # query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
        #         time_frame + '&sort=' + query_sort_by + '&order=desc'

        # Create search query
        query = query_search_terms + '+' + query_search_locations + '+language:python+' + \
                time_frame + '&sort=' + query_sort_by + '&order=desc'

        # Create initial url from query
        url = 'https://api.github.com/search/repositories?page=1&per_page=' + str(n_results_per_page) + '&q=' + query

        # Submit request and time response
        begin_query = time.time()
        response = requests.get(url, headers=headers)
        repo_count = json.loads(response.text).get('total_count')
        # print('\nNumber of Repos: %d' % repo_count)

        # JOB: Check if query yields more than limit of 1,000 search results
        # If number of found repositories exceeds limit of 1000 search results, reduce period_length
        slack = 0.8  # Factor by which to reduce period length as a security margin
        if repo_count > 1000:
            delta_period_length = datetime.timedelta(
                days=floor(((1000 / repo_count) * period_length.days * slack)))  # Calculate delta for period length
            last_date = last_date - delta_period_length  # Calculate modified last_date
            period_length -= delta_period_length  # Modify period_length for search
            print('Change of period length to: %d' % period_length.days)
            continue
        else:
            pass

        # JOB: Process response for initial search results page
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

        # JOB: Iterate over remaining search results pages
        if response.status_code == 200:
            # Iterate through returned pages
            iterate_pages(response, tokens)  # Call function to iterate through paginated results

        else:
            # If request unsuccessful, print error message
            print('Request failed. Error code: %d - %s' % (response.status_code, response.reason))

        # Set search date window to next time frame
        search_from_date = last_date + datetime.timedelta(days=1)

        # JOB: Check if end of search time frame is reached
        if last_date == search_end_date:
            break  # End search iterations
        elif last_date + period_length > search_end_date:
            last_date = search_end_date  # Restrict search time frame to past
        else:
            last_date += period_length  # If end not reached, increase last date by period length

    # Save json file locally in specified location
    # with open('Repo_Search_Results.json', 'w', encoding='utf8') as json_file:
    #     json.dump(repo_list, json_file)

    # Print number of saved repositories
    repo_count = len(repo_list)
    print('\n\nNumber of saved repositories: %d' % repo_count)
    print('\nExtracting meta data from found repositories:')

    # Initialize data list for json export
    data = []

    # JOB: Iterate over found repositories, create repository items and add to data list
    for repo_index, repo in enumerate(repo_list):
        # Seach for 'save' in .py file
        # query_url = 'https://api.github.com/search/code?q=save+extension:py+repo:' + repo['full_name']

        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1

        # Specify base url for repository search
        url = 'https://api.github.com/repos/'

        # Specify path to readme file
        readme_path = url + repo['full_name'] + '/readme'

        # Specify request header consisting of authorization and accept string
        headers = {'Authorization': 'token ' + tokens[token_index],
                   'Accept': 'application/vnd.github.com.v3.html'}

        # Time begin of query
        begin_query = time.time()

        # Query API for readme file
        response = requests.get(readme_path, headers=headers)

        # JOB: Get description language
        description_language = identify_language(repo['description'])

        # JOB: Get Readme text and language
        # Hand response to markdown parser
        readme_text = get_readme(response)
        has_readme = True if readme_text is not None else False
        readme_language = identify_language(readme_text)

        # JOB: Get repository tags
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1  # Increase counter
        # Specify header
        headers = {'Authorization': 'token ' + tokens[token_index],
                   'Accept': 'application/vnd.github.mercy-preview+json'}
        # Specify path to labels
        tags_path = url + repo['full_name'] + '/topics'
        # Query API for labels
        response = requests.get(tags_path, headers=headers)
        # Retrieve tags
        tags_list = json.loads(response.text).get('names')

        # JOB: Search for h5 files
        token_index = token_counter % rotation_cycle
        token_counter += 1  # Increase counter
        headers = {'Authorization': 'token ' + tokens[token_index]}

        # Specify search for file extensions '.h5' and 'hdf5'
        query_url = 'https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:' + repo['full_name']
        response = requests.get(query_url, headers=headers)

        try:
            print('\n\nRemaining/Limit for token %d: %d/%d' % (token_index,
                                                               int(response.headers['X-RateLimit-Remaining']),
                                                               int(response.headers['X-RateLimit-Limit'])))
            if int(response.headers['X-RateLimit-Remaining'] <= 3):
                time.sleep(2)
                print('Execution paused for 2 seconds.')
            else:
                pass
        except KeyError as e:
            print('Error retrieving X-RateLimit: %s' % e.args)

        has_h5_file = False

        # Check for response
        if response.status_code == 200:
            # If request to API successful
            has_h5_file = json.loads(response.text).get('total_count') > 0  # '.h5' or '.hdf5' file found in repo

        # JOB: Save meta data to item dict
        # Specify repo meta data to be extracted from API response
        item = {'repo_url': repo['html_url'],
                'repo_name': repo['name'],
                'repo_full_name': repo['full_name'],
                'repo_owner': repo['owner']['login'],
                'repo_desc': repo['description'],
                'description_language': description_language,
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
                'github_id': repo['id'],
                'is_fork': repo['fork'],
                'readme_text': readme_text,
                'has_readme': has_readme,
                'readme_language': readme_language,
                'repo_tags': tags_list,
                'has_h5': has_h5_file
                }

        # # Add architecture attribute
        # if has_architecture:
        #     item['has_structure'] = True
        #     # print(json.loads(response.text).get('items')[0])
        # else:
        #     item['has_structure'] = False

        # Append item to data list
        data.append(item)

        # Time end of request
        end_time = time.time()
        time_diff = end_time - begin_query

        # Print search progress as progress bar
        print_progress(repo_index + 1, len(repo_list), prog='Last request: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)

    # Create MongoDB collection instance
    collection = DataCollection.DataCollection('Repos_Exp')
    # print(collection)

    # Insert data into database if list is not empty
    if data:
        collection.collection_object.insert_many(data)

    # Check database size and print to console
    print('\n\nNumber of Repositories in Database: %d' % collection.collection_object.count_documents({}))


if __name__ == '__main__':
    # Specify start and end search dates
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2019, 5, 25)

    periods = list(split_time_interval(start, end, 15))

    for tf in periods:
        print('Current time frame: %s - %s' % (tf[0], tf[1]))
        # Retrieve token lists
        token_lists = get_access_tokens()

        time_frames = list(split_time_interval(tf[0], tf[1], 3))
        print(', '.join(str(f[0]) + ' - ' + str(f[1]) for f in time_frames))

        processes = []
        for index in range(3):
            print('Start process %d' % (index + 1))
            p = Process(target=seach_repos, args=(*time_frames[index], token_lists[0]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
