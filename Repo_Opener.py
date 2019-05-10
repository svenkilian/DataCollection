import pymongo
import pprint
import webbrowser

from pymongo import MongoClient

if __name__ == '__main__':

    repo_links = []

    client = MongoClient('localhost')

    db = client['GitHub']
    repos = db['repos']

    for repo in repos.find():
        # pprint.pprint(repo['page'])
        repo_links.append(repo['page'])

    for link in repo_links[:10]:
        webbrowser.open_new_tab(link)
