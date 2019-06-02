"""
This class extracts the repository README.md and saves the plain text in the database
"""
from config import ROOT_DIR
import os
import json
import time

import pymongo
import requests
from bs4 import BeautifulSoup
from bson import ObjectId
from bson.json_util import dumps
from markdown import markdown
from pandas import read_json
import urllib3
import re
from Helper_Functions import print_progress


def get_readme(response):
    """
    Get plain text from readme file
    :param response: Res
    :return: Plain plain_text of readme file
    """
    plain_text = None
    link_list = []
    reference_list = []
    # Request successful
    if response.status_code == 200:

        # Extract plain_text and remove all line breaks
        soup = BeautifulSoup(response.text, features='lxml')

        # Find all arxiv links and append to reference_list
        for reference in soup.findAll('a', attrs={'href': re.compile('(arxiv|ieee.org)')}):
            reference_list.append(reference.get('href'))

        # Find all links and append to link_list
        for link in soup.findAll('a', attrs={'href': re.compile('^http://')}):
            link_list.append(link.get('href'))

        # Remove references from link_list
        link_list = list(set(link_list) - set(reference_list))

        plain_text = ''.join(soup.find_all(text=True))
        plain_text = plain_text.replace('\n', ' ').replace('\t', ' ').replace('\r', '')

        # Set plain_text to null if empty string
        if plain_text == '':
            plain_text = None

    # Request unsuccessful
    elif response.status_code == 404:
        pass
        # print(' - Repository without readme found for: %s\n%s' % (response.plain_text, response.request.url))

    else:
        print('Unknown error occurred while parsing readme file: %d' % response.status_code)
        print(response.reason)
        if response.status_code == 403:
            print('Access denied.')

    # print('\n\nLimit: %d' % int(response.headers['X-RateLimit-Limit']))
    # print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))

    return plain_text, link_list, reference_list


def set_full_name(repos):
    """
    -- Deprecated --
    :param repos:
    :return:
    """
    # Start timer
    start_time = time.time()

    # Iterate through repository list
    for index, repo in repos.iterrows():
        begin_time = time.time()

        repo_id = repo['_id'].get('$oid')
        collection.find_one_and_update(
            {'_id': ObjectId(repo_id)},
            {'$set': {'repo_full_name': repo['repo_name']}}
        )

        end_time = time.time()
        time_diff = end_time - begin_time

        # Print search progress as progress bar
        print_progress(index + 1, len(repos), prog='Last DB write: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)


def add_data_field(repos, access_path):
    """
    -- Deprecated --
    :param repos:
    :param access_path:
    :return:
    """
    url = 'https://api.github.com/repos/'
    # Start timer
    start_time = time.time()

    # Retrieve local access token for GitHub API access
    with open(access_path, 'r') as f:
        access_tokens = f.readlines()
    # List tokens
    access_tokens = [token.rstrip('\n') for token in access_tokens]
    # Determine length of rotation cycle and set counter to 0
    rotation_cycle = len(access_tokens)
    token_counter = 0

    # Iterate through repository list
    for index, repo in repos.iterrows():
        begin_time = time.time()
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1
        # Set header
        headers = {'Authorization': 'token ' + access_tokens[token_index]}
        # Build repo URL
        repo_url = url + repo['repo_name']

        # Send request
        try:
            response = requests.get(repo_url, headers=headers)
        except Exception as e:
            print('Encountered Exception: %s' % e.args[0])

        if response.status_code == 200:
            # If request successful

            # Create json file from response
            repo_instance = json.loads(response.text)

            meta_data = {'private': repo_instance['private'],
                         'repo_created_at': repo_instance['created_at'],
                         'homepage': repo_instance['homepage'],
                         'size': repo_instance['size'],
                         'language': repo_instance['language'],
                         'has_wiki': repo_instance['has_wiki'],
                         'license': repo_instance['license'],
                         'open_issues_count': repo_instance['open_issues_count'],
                         'subscribers_count': repo_instance['subscribers_count'],
                         'github_id': repo_instance['id'],
                         'is_fork': repo_instance['fork'],
                         'repo_full_name': repo_instance['full_name']}

            repo_id = repo['_id'].get('$oid')
            collection.find_one_and_update(
                {'_id': ObjectId(repo_id)},
                {'$set': meta_data}
            )

        end_time = time.time()
        time_diff = end_time - begin_time

        # Print search progress as progress bar
        print_progress(index + 1, len(repos), prog='Last DB write: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)


if __name__ == '__main__':
    # Retrieve database credentials
    cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials/connection_creds.txt')
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    # Establish database connection
    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.Repositories
    print(client.server_info())

    # Check database size and print to console
    print('Number of Repositories in Database: %d' % collection.count_documents({}))

    # Save database query result to json
    data = dumps(collection.find({'readme_text': None}, projection=['repo_full_name']))

    # Make DataFrame from json
    data_frame = read_json(data)
    print(data_frame)

    # # Extract repository name from URL and append as new column
    # data_frame['repo_name'] = data_frame['repo_url'].apply(lambda x: '/'.join(x.split('/')[-2:]))

    # Save all repository names in list
    repos = data_frame[['repo_full_name', '_id']]
    # print(repos.head())
    # print('Number of Repositories without Readme file: %d' % repos.shape[0])
    #
    # batch_1 = repos[:int(repos.shape[0] / 2)]
    # print('Items in first batch: %d' % batch_1.shape[0])
    # batch_2 = repos[int(repos.shape[0] / 2):]
    # print('Items in second batch: %d' % batch_2.shape[0])
    cred_path_1 = os.path.join(ROOT_DIR + '/DataCollection/credentials/GitHub_Access_Token.txt')
    # cred_path_2 = os.path.join(ROOT_DIR + '/DataCollection/credentials/GitHub_Access_Token_2.txt')
    #
    # thread_1 = threading.Thread(target=get_readme, args=(batch_1,))
    # thread_2 = threading.Thread(target=get_readme, args=(batch_2,))
    #
    # thread_1.start()
    # thread_2.start()

    get_readme(repos, access_path=cred_path_1, collection=collection)
