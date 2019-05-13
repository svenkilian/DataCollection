# This class implements database access to the MongoDB Atlas web database

from bson import json_util
import json
import pandas as pd
import pymongo

if __name__ == "__main__":
    # Establish database connection
    cred_path = 'C:/Users/svenk/PycharmProjects/GitHub_Scraping/connection_creds.txt'
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.repos
    # print(json.dumps(client.server_info(), default=json_util.default, sort_keys=True, indent=3))
