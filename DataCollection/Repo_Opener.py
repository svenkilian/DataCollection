import pymongo
import os
import pprint
import webbrowser
import pandas as pd
from config import ROOT_DIR
from pymongo import MongoClient
from Readme_Scraper import add_data_field

if __name__ == '__main__':
    repo_links = []
    cred_path = os.path.join(ROOT_DIR, 'DataCollection\credentials\connection_creds.txt')

    with open(cred_path, 'r') as f:
        connection_string = f.read()

    client = pymongo.MongoClient(connection_string)

    collection = client.GitHub.Repositories

    # for repo in collection.find():
    #     # pprint.pprint(repo['page'])
    #     repo_links.append(repo['repo_url'])
    #
    # for link in repo_links[:10]:
    #     webbrowser.open_new_tab(link)

    data = pd.DataFrame(list(collection.find()))

    # print(data['repo_desc'].str.contains('keras', regex=False).sum())
    # print(data['readme_text'].str.contains('keras', regex=False).sum())
    # print(data['repo_name'].str.contains('keras', regex=False).sum())

    print(data[data['size'].isna()]['repo_full_name'].tolist())

    data = data[data['size'].isna()][['repo_full_name', '_id']]
    data.rename(columns={'repo_full_name': 'repo_name'}, inplace=True)
    deprecated = data['_id'].tolist()

    print(deprecated)
    collection.remove({'_id': {'$in': deprecated}})
    # cred_path = './credentials/connection_creds.txt'
    # add_data_field(data, cred_path)
