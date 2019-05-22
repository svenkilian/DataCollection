from config import ROOT_DIR
import os
import pymongo
from bson import json_util
import json


class DataCollection:
    """
    Implements a MongoDB Atlas Database Collection
    """

    def __init__(self, repo_name):
        """
        Constructor for a Collection object holding the reference to MongoDB collection object
        :param repo_name: name of repository within database
        """

        # Establish database connection
        self.cred_path = os.path.join(ROOT_DIR,
                                      'DataCollection/credentials/connection_creds.txt')  # Path to database login credentials

        # Read connection string from text file
        with open(self.cred_path, 'r') as f:
            connection_string = f.read()

        # Instantiate MongoClient
        self.client = pymongo.MongoClient(connection_string)
        self.collection_object = self.client.GitHub[repo_name]

    def __repr__(self):
        """
        Specify string representation of DataCollection instance
        :return: string representation of DataCollection instance
        """

        # Return collection
        return 'Current collection: %s' % self.collection_object.full_name

    def delete_duplicates(self):
        """
        Delete duplicates from collection based on repo_url
        :return:
        """
        cursor = self.collection_object.aggregate(
            [
                {"$group": {"_id": "$repo_url", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
                {"$match": {"count": {"$gte": 2}}}
            ]
        )

        n_docs = self.collection_object.count_documents({})
        response = []
        for doc in cursor:
            del doc["unique_ids"][0]
            for id in doc["unique_ids"]:
                response.append(id)

        n_duplicates = len(response)
        self.collection_object.remove({"_id": {"$in": response}})

        print('Number of entries in database before deletion: %d' % n_docs)
        print('Number of duplicate entries before deletion: %d \n' % n_duplicates)
        print('Number of entries in database after deletion of duplicates: %d' %
              self.collection_object.count_documents({}))
        print('Number of entries with structure information: %d \n\n' %
              self.collection_object.count_documents({'has_structure': True}))

    def count_duplicates(self):
        """
        Counts duplicate entries in collection by repo_url
        :return:
            n_docs: total number of documents in collection
            n_structure: number of documents with structure information
            n_duplicates: number of duplicate documents in collection

        """
        cursor = self.collection_object.aggregate(
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

        n_docs = self.collection_object.count_documents({})
        n_structure = self.collection_object.count_documents({'has_structure': True})
        n_duplicates = len(response)

        print('Number of entries in database: %d' % n_docs)
        print('Number of entries with structure information: %d' % n_structure)
        print('Number of duplicates in data base: %d \n' % n_duplicates)

        return n_docs, n_structure, n_duplicates

    def clear_all_entries(self):
        print('Number of entries before reset: %d' % self.collection_object.count_documents({}))
        self.collection_object.remove({})
        print('Number of entries after reset: %d' % self.collection_object.count_documents({}))


if __name__ == "__main__":
    collection = DataCollection('Repositories')

    # Print database server info
    print(collection)

    print('Number of entries in database: %d' % collection.collection_object.count_documents({}))
    print('Number of entries with structure information: %d' %
          collection.collection_object.count_documents({'has_structure': True}))

    # collection.delete_duplicates()
    # collection.clear_all_entries()  # Use to clear all entries
    collection.count_duplicates()
