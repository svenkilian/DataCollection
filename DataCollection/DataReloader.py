"""
This module implements a method to collect additional data from GitHub.
"""

import multiprocessing
from itertools import repeat
from multiprocessing import Pool

import math
import requests

from DataCollection import DataCollection
from HelperFunctions import *
import config
import sys
import numpy as np

token_counter = 0

# Get token list
tokens = get_access_tokens()[0]
token_cycle = len(tokens)


def retrieve_attributes(attributes, repo):
    """
    Retrieves additional GitHub repository attributes.

    :param attributes: Attributes to retrieve from GitHub repository
    :param repo: Repository name to retrieve attributes from
    :return: List of repository _id, repo_full_name and retrieved attributes
    """

    global token_counter, token_cycle, tokens

    api_url = 'https://api.github.com/repos/' + repo[1]
    token_index = token_counter % token_cycle
    token_counter += 1
    headers = {'Authorization': 'token ' + tokens[token_index],
               'Accept': 'application/vnd.github.mercy-preview+json'}

    response = requests.get(api_url, headers=headers)

    if token_counter % 100 == 0:
        check_access_tokens(token_index, response)

    response_data = json.loads(response.content)

    attribute_values = [response_data.get(key) for key in attributes]

    return [*repo, *attribute_values]


if __name__ == '__main__':
    """
    Main method
    """

    collection = DataCollection.DataCollection('Repos_New')

    # Define attributes to retrieve
    attributes = ['subscribers_count']

    # Get list of repositories
    data_full = pd.read_json(os.path.join(ROOT_DIR, 'DataCollection/data/data.json'))
    repo_list = data_full[['_id', 'repo_full_name']].values
    repo_list = repo_list[4500:]

    print('Loaded data')

    process_pool = Pool(processes=multiprocessing.cpu_count())

    start_time = time.time()
    attribute_mapping = np.asarray(
        process_pool.starmap(retrieve_attributes, zip(repeat(attributes), repo_list)))

    process_pool.close()
    process_pool.join()

    df = pd.DataFrame.from_records(columns=['_id', 'repo_full_name', *attributes], data=attribute_mapping,
                                   index=['repo_full_name'])

    df['subscribers_count'] = df['subscribers_count'].astype('float')

    collection.add_many(df['_id'].values, 'subscribers_count', df['subscribers_count'].values)

    end_time = time.time()

    print('Duration: %g seconds' % round(end_time - start_time, 2))
