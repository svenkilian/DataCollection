"""
This class extracts the repository README.md and saves the plain text in the database
"""

import time

import pymongo
import requests
from bs4 import BeautifulSoup
from bson import ObjectId
from bson.json_util import dumps
from markdown import markdown
from pandas import read_json

from DataCollection.Helper_Functions import print_progress

if __name__ == '__main__':
    # Retrieve database credentials
    cred_path = './credentials/connection_creds.txt'
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    # Establish database connection
    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.Repositories
    print(client.server_info())

    # Check database size and print to console
    print('Number of Repositories in Database: %d' % collection.count_documents({}))

    # Save database query result to json
    data = dumps(collection.find(projection=['repo_url']))

    # Make DataFrame from json
    data_frame = read_json(data)

    # Extract repository name from URL and append as new column
    data_frame['repo_name'] = data_frame['repo_url'].apply(lambda x: '/'.join(x.split('/')[-2:]))

    # Save all repository names in list
    repos = data_frame[['repo_name', '_id']][:]

    start_time = time.time()

    for index, repo in repos.iterrows():
        begin_time = time.time()
        # Specify path to readme file
        readme_path = 'https://raw.githubusercontent.com/' + repo['repo_name'] + '/master/README.md'

        # Query API for readme file
        response = requests.get(readme_path)

        text = ''

        # Request successful
        if response.status_code == 200:
            # Convert markdown to html
            text = markdown(response.text)

            # Extract text and remove all line breaks
            text = ''.join(BeautifulSoup(text, features='lxml').find_all(text=True))
            text = text.replace('\n', ' ').replace('\t', ' ').replace('\t', '')

        # Request unsuccessful
        elif response.status_code == 404:
            text = None

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
        print_progress(index + 1, len(repos), prog='Last DB write: %g' % round(time_diff, 2),
                       time_lapsed=end_time - start_time)
