import pymongo
import pprint
import webbrowser
import pandas as pd

from pymongo import MongoClient

if __name__ == '__main__':

    repo_links = []

    cred_path = 'C:/Users/svenk/PycharmProjects/GitHub_Scraping/connection_creds.txt'
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

    print(data['repo_desc'].str.contains('keras', regex=False).sum())
    print(data['readme_text'].str.contains('keras', regex=False).sum())