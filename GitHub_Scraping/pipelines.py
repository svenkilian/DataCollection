# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymongo
from scrapy.conf import settings
from scrapy.exceptions import DropItem
from scrapy import log
import os.path


class GithubScrapingPipeline(object):

    def __init__(self):
        # connection = pymongo.MongoClient(
        #     settings['MONGODB_SERVER'],
        #     settings['MONGODB_PORT']
        # )
        # db = connection[settings['MONGODB_DB']]
        # self.collection = db[settings['MONGODB_COLLECTION']]

        cred_path = 'C:/Users/svenk/PycharmProjects/DataCollection/connection_creds.txt'
        print(cred_path)

        with open(cred_path, 'r') as f:
            connection_string = f.read()

        client = pymongo.MongoClient(connection_string)

        self.collection = client.GitHub.repos
        print(client.server_info())

    def process_item(self, item, spider):
        self.collection.insert(dict(item))
        log.msg('Object added to Database',
                level=log.DEBUG, spider=spider)
        print('Object added to Database \n')

        return item
