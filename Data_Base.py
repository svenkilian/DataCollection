# This class implements database access to the MongoDB Atlas web database

from bson import json_util
import json
# import pandas as pd
import pymongo


def delete_duplicates(coll):
    cursor = coll.aggregate(
        [
            {"$group": {"_id": "$repo_url", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gte": 2}}}
        ]
    )

    n_docs = coll.count_documents({})
    response = []
    for doc in cursor:
        del doc["unique_ids"][0]
        for id in doc["unique_ids"]:
            response.append(id)

    n_duplicates = len(response)
    coll.remove({"_id": {"$in": response}})

    print('Number of entries in database before deletion: %d' % n_docs)
    print('Number of duplicate entries before deletion: %d \n' % n_duplicates)
    print('Number of entries in database after deletion of duplicates: %d' % collection.count_documents({}))
    print('Number of entries with structure information: %d \n\n' % collection.count_documents({'has_structure': True}))


def count_duplicates(coll):
    cursor = coll.aggregate(
        [
            {"$group": {"_id": "$repo_url", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gte": 2}}}
        ]
    )

    response = []
    for doc in cursor:
        del doc["unique_ids"][0]
        for id in doc["unique_ids"]:
            response.append(id)

    n_docs = collection.count_documents({})
    n_structure = collection.count_documents({'has_structure': True})
    n_duplicates = len(response)

    print('Number of entries in database: %d' % n_docs)
    print('Number of entries with structure information: %d' % n_structure)
    print('Number of duplicates in data base: %d \n' % n_duplicates)

    return n_docs, n_structure, n_duplicates


def clear_all_entries(coll):
    print('Number of entries before reset: %d' % collection.count_documents({}))
    coll.remove({})
    print('Number of entries after reset: %d' % collection.count_documents({}))


if __name__ == "__main__":
    # Establish database connection
    cred_path = 'connection_creds.txt'
    with open(cred_path, 'r') as f:
        connection_string = f.read()

    client = pymongo.MongoClient(connection_string)
    collection = client.GitHub.repos

    # Print database server info
    # print(json.dumps(client.server_info(), default=json_util.default, sort_keys=True, indent=3))

    # print('Number of entries in database: %d' % collection.count_documents({}))
    # print('Number of entries with structure information: %d' % collection.count_documents({'has_structure': True}))

    delete_duplicates(collection)
    # clear_all_entries(collection)
    count_duplicates(collection)
