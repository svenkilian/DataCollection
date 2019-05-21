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
import threading
from docutils.core import publish_parts

from Helper_Functions import print_progress


def parse_readme_to_text(response):
    # Request successful
    if response.status_code == 200:
        # Convert markdown to html
        text = markdown(response.text)
        # Extract text and remove all line breaks
        text = ''.join(BeautifulSoup(text, features='lxml').find_all(text=True))
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\t', '')

    # Request unsuccessful
    elif response.status_code == 404:
        text = publish_parts(response.text, writer_name='html')['html_body']
        # Extract text and remove all line breaks
        text = ''.join(BeautifulSoup(text, features='lxml').find_all(text=True))
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\t', '')

        print(' - Response status code 404: Page not found')

    else:
        text = None
        print(' - Unexpected response code: %d' % response.status_code)

    return text


def get_readme(repos, access_path):
    # Start timer
    start_time = time.time()
    counter = 1

    # Retrieve local access token for GitHub API access
    with open(access_path, 'r') as f:
        access_tokens = f.readlines()
    # List tokens
    access_tokens = [token.rstrip('\n') for token in access_tokens]
    # Determine length of rotation cycle and set counter to 0
    rotation_cycle = len(access_tokens)
    token_counter = 0

    url = 'https://api.github.com/repos/'

    # Iterate through repository list
    for index, repo in repos.iterrows():
        begin_time = time.time()
        # Determine current token index and increment counter
        token_index = token_counter % rotation_cycle
        token_counter += 1

        # Specify path to readme file
        readme_path = url + repo['repo_full_name'] + '/readme'

        # Specify request header consisting of authorization and accept string
        headers = {'Authorization': 'token ' + access_tokens[token_index],
                   'Accept': 'application/vnd.github.com.v3.raw'}

        # Query API for readme file
        response = requests.get(readme_path, headers=headers)

        text = parse_readme_to_text(response)

        # print(text)
        # print(repo['_id'])

        repo_id = repo['_id'].get('$oid')
        collection.find_one_and_update(
            {'_id': ObjectId(repo_id)},
            {'$set': {'readme_text': text}}
        )

        end_time = time.time()
        time_diff = end_time - begin_time

        # Print search progress as progress bar
        print_progress(counter, len(repos), prog='Last DB write: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)

        counter += 1


def set_full_name(repos):
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

    get_readme(repos, access_path=cred_path_1)
