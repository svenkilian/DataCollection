# This module implements database access to GitHub's REST-API for search queries
import json
from multiprocessing import Process

import requests
from math import ceil

from DataCollection import DataCollection
from HelperFunctions import *


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
    """
    Finds repositories between start_date and end_date mathing the specified search word in the specified locations.

    :param start_date: Start date of search
    :param end_date: End date of search
    :param tokens: Authentication tokens to use
    :return:
    """
    global token_counter, rotation_cycle, page_counter, n_results, start_time, repo_list
    # JOB: Specify search terms, time period to search, resources to search
    # Configure number and size of requests (responses are paginated)
    n_results_per_page = 100

    # Initialize repository list
    repo_list = []

    # Search query string, see https://developer.github.com/v3/search/#search-repositories for documentation
    # Example search: https://api.github.com/search/code?q=extension:h5+extension:hdf5+repo:GilbertoEspinoza/emojify
    search_terms = ['keras']
    query_search_terms = '+'.join(search_terms)

    # Specify words that are excluded from search
    # stop_words = ['tutorial']
    # query_stop_words = 'NOT+' + '+NOT+'.join(stop_words)

    search_locations = ['title', 'readme', 'description']  # Specify locations to search
    query_search_locations = '+'.join(['in:' + location for location in search_locations])

    # Set search time interval
    # Memo: Keras release date: 2015-03-27
    search_from_date = start_date
    search_end_date = end_date

    # Specify sorting of results - Irrelevant for comprehensive search
    query_sort_by = 'score'  # updated, stars, forks, default: score
    # query = query_search_terms + '+' + query_stop_words + '+' + query_search_locations + '+language:python+' + \
    #         query_search_from + '&sort=' + query_sort_by + '&order=desc'

    # Specify search string for search time frame
    time_frame = 'created:' + search_from_date.isoformat() + '..' + search_end_date.isoformat()

    # Create full query
    query = query_search_terms + '+' + query_search_locations + '+language:python+' + \
            time_frame + '&sort=' + query_sort_by + '&order=desc'

    # Set rotation cycle to number of tokens being rotated
    rotation_cycle = len(tokens)

    # JOB: Create initial query and send request
    # Create initial url from query
    url = 'https://api.github.com/search/repositories?page=1&per_page=' + str(n_results_per_page) + '&q=' + query

    # Set results number and token counter to zero
    n_results = 0
    page_counter = 0
    token_counter = 0

    # Specify header with authentication for first query using first token in list
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
    else:
        # In case of unsuccessful request, terminate program
        print('Initial request failed. Terminating processes.')
        sys.exit(0)  # Exit program

    end_time = time.time()  # Stop timer
    time_diff = end_time - begin_query  # Calculate duration of query response

    # Calculate total time period to search
    time_delta = (search_end_date - search_from_date).days + 1  # Search time period length in days

    # Calculate search period to use for iteration of results
    avg_daily_results = n_results / time_delta  # Average number of search results per day

    avg_period = 1000 / avg_daily_results  # Average period length yielding 1000 search results (API limit)

    # Set length for search period to average period yielding 1000 results
    period_length = datetime.timedelta(days=avg_period)
    # print('Period length (days): %d\n\n' % period_length.days)

    # Specify last date for current search iteration
    last_date = search_from_date + period_length

    # If last date is in the future, set last_date to end_date
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
            period_length = datetime.timedelta(
                days=floor(((1000 / repo_count) * period_length.days * slack)))  # Calculate delta for period length
            last_date = search_from_date + period_length  # Calculate modified last_date
            print('\nChange of period length to: %d' % period_length.days)
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
        time_diff = end_time - begin_query

        # Print search progress as progress bar
        print_progress(page_counter, ceil(n_results / 100.0),
                       prog='Last request: %g' % round(time_diff, 2),
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

        # JOB: Set search date window to next time frame
        # Set search from date to one day after previous last_date
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
    print('Finished collecting search results.')
    print('\n\nNumber of saved repositories: %d' % repo_count)
    print('\nExtracting meta data from found repositories:')

    # Initialize data list for json export
    data = []

    # JOB: Iterate over found repositories, create repository items and add to data list
    for repo_index, repo in enumerate(repo_list):
        # Seach for 'save' in .py file
        # query_url = 'https://api.github.com/search/code?q=save+extension:py+repo:' + repo['full_name']

        # JOB: Get description language
        description_language = identify_language(repo['description'])

        # JOB: Extract plain text from readme files
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

        # JOB: Get Readme text, language and hyperlinks
        # Hand response to markdown parser
        readme_text, link_list, reference_list = extract_from_readme(response)
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

        check_access_tokens(token_index, response)

        # Initialize empty list of links
        h5_files_links = []

        # Check for response
        if response.status_code == 200:
            # If request to API successful
            has_h5_file = json.loads(response.text).get('total_count') > 0  # '.h5' or '.hdf5' file found in repo

            if has_h5_file:
                h5_files = json.loads(response.text).get('items')
                for file in h5_files:
                    h5_files_links.append(file['html_url'])

        else:
            # raise Exception('Could not check for h5 files')
            pass

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
                'has_h5': has_h5_file,
                'h5_files_links': h5_files_links,
                'see_also_links': link_list,
                'reference_list': reference_list,
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
    collection = DataCollection('Exper')
    # print(collection)

    # Insert data into database if list is not empty
    if data:
        collection.collection_object.insert_many(data)

    # Check database size and print to console
    print('\n\nNumber of Repositories in Database: %d' % collection.collection_object.count_documents({}))


if __name__ == '__main__':
    """
    Main method to be called when API_Access module is run.
    """

    # Specify start and end search dates
    start = datetime.date(2019, 5, 20)  # Letzter Stand: 2018, 12, 1 - 2018, 12, 31
    end = datetime.date(2019, 5, 31)
    n_days = (end - start).days + 1
    n_macro_periods = 1
    print('Searching for repositories between %s and %s\n'
          'Partionioning into %d macro periods\n\n' % (start.isoformat(), end.isoformat(), n_macro_periods))

    if n_macro_periods > n_days:
        # If trying to split up time frame into more time periods than days, limit to n_days
        n_macro_periods = n_days
        print('Macro time periods: %d' % n_macro_periods)

    # Get macro time periods
    periods = list(split_time_interval(start, end, n_macro_periods, n_days))

    print('Macro periods:')
    print(', '.join(str(f[0]) + ' - ' + str(f[1]) for f in periods))
    print()

    for tf in periods:
        n_process = 4  # Specify number of parallel processes to be used
        print('\nCurrent time frame: %s - %s' % (tf[0], tf[1]))
        # Retrieve token lists
        token_lists = get_access_tokens()

        # Get list of time frames per period

        # Number of days in sub period
        n_days_micro = (tf[1] - tf[0]).days + 1

        if n_days_micro < n_process:
            # Limit number of processes
            n_process = n_days_micro

        time_frames = list(split_time_interval(tf[0], tf[1], n_process, n_days_micro))
        print('Processing time periods in parallel:')
        print(', '.join(str(f[0]) + ' - ' + str(f[1]) for f in time_frames))
        print()

        # Define empty list for processes
        processes = []
        # JOB: Initiate and start processes
        for index in range(n_process):
            print('Start process %d' % (index + 1))
            p = Process(target=seach_repos, args=(*time_frames[index], token_lists[0]))
            processes.append(p)
            p.start()

        for p in processes:
            # Join processes with main process
            p.join()
