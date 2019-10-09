"""
This class implements a MongoDB Atlas Database Collection wrapper.
"""

from bson import ObjectId

from config import ROOT_DIR
import os
import pymongo


class DataCollection:
    """
    Implements a MongoDB Atlas Database Collection wrapper class
    """

    def __init__(self, collection_name):
        """
        Constructor for a Collection object holding the reference to MongoDB collection object

        :param repo_name: name of collection within database
        """

        # Establish database connection
        self.cred_path = os.path.join(ROOT_DIR,
                                      'DataCollection/credentials/connection_creds.txt')

        # Read connection string from text file
        with open(self.cred_path, 'r') as f:
            connection_string = f.read()

        # Instantiate MongoClient
        self.client = pymongo.MongoClient(connection_string)
        self.collection_object = self.client.GitHub[collection_name]

    def __repr__(self):
        """
        Specifies string representation of DataCollection instance

        :return: string representation of DataCollection instance
        """

        # Return collection
        return 'Current collection: %s' % self.collection_object.full_name

    def delete_duplicates(self):
        """
        Deletes duplicates from collection based on repo_url

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

    def count_attribute(self, attribute):
        """
        Counts number of times the given attribute is true.

        :param attribute: Attribute to count
        :return: Number of true attribute occurrences
        """

        attribute_count = self.collection_object.find({attribute: True}).count()

        print('Number of true %s: %d \n' % (attribute, attribute_count))

        return attribute_count

    def count_empty_array_attributes(self, attribute):
        """
        Counts number of times the given attribute is true.

        :param attribute: Attribute to count
        :return: Number of true attribute occurrences
        """

        attribute_count = self.collection_object.find({"$or": [{attribute: {"$size": 0}}, {attribute: None}]}).count()

        print('Number of true %s: %d \n' % (attribute, attribute_count))

        return attribute_count

    def add_attribute(self, id, attribute_name, value):
        """
        Adds attribute attribute_name with specified value to specified document.

        :param id: Document id to add attribute to
        :param attribute_name: Name of attribute to add
        :param value: Value to assign to attribute
        :return:
        """

        self.collection_object.update_one({
            '_id': ObjectId(id['$oid'])
        }, {
            '$set': {
                attribute_name: value
            }
        }, upsert=False)

    def add_many(self, ids, attribute_name, values):
        """
        Adds attribute and corresponding value to list of documents.

        :param ids: List of document IDs to add attribute to
        :param attribute_name: Attribute name to add/update
        :param values: Value to assign to attribute
        :return:
        """

        for id, value in zip(ids, values):
            self.add_attribute(id, attribute_name, value)

    def clear_all_entries(self):
        """
        Clears all documents in cloud data base

        :return:
        """
        print('Number of entries before reset: %d' % self.collection_object.count_documents({}))
        self.collection_object.remove({})
        print('Number of entries after reset: %d' % self.collection_object.count_documents({}))

    def copy_to_collection(self, destination):
        """
        Copies calling collection to destination collection.

        :param destination: Existing destination collection
        :return:
        """
        pipeline = [{'$match': {}},
                    {'$out': destination}]

        self.collection_object.aggregate(pipeline)


if __name__ == "__main__":
    collection = DataCollection('Repos_Exp')

    # Print database server info
    print(collection)

    print('Number of entries in database: %d' % collection.collection_object.count_documents({}))

    # collection.delete_duplicates()
    # collection.clear_all_entries()  # Use to clear all entries
    # collection.count_duplicates()
    collection.copy_to_collection('Repos_New')

    n_docs = collection.collection_object.count_documents({})

    n_keras_used = collection.count_attribute('keras_used')
    n_structure_h5 = collection.count_attribute('h5_data.extracted_architecture')
    n_structure_py = collection.count_attribute('py_data.model_file_found')
    n_empty_tags = collection.count_empty_array_attributes('repo_tags')

    print('Number of entries in database: %d' % n_docs)
    print('Number of entries with structure information from h5: %d' % n_structure_h5)
    print('Number of entries with structure information from .py file: %d \n' % n_structure_py)
    print('Total number of entries with structure information: %d \n' % (n_structure_h5 + n_structure_py))
    print('Total number of entries without topic tags: %d \n' % n_empty_tags)
    try:
        print('Percentage of entries with structure information: %g \n' % ((n_structure_h5 + n_structure_py) / n_docs))
        print('Percentage of entries using Keras: %g \n' % (n_keras_used / n_docs))
    except ZeroDivisionError:
        print('No documents in database.')
