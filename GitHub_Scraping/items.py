# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class GithubScrapingItem(Item):
    page = Field()
    repo_name = Field()
    repo_desc = Field()
    repo_link = Field()
    repo_owner = Field()
    repo_ref = Field()
    repo_dcount = Field()
    repo_last_mod = Field()
    repo_watch = Field()
    repo_stars = Field()
    repo_forks = Field()


class GitHubRepositoryItem(Item):
    repo_name = Field()
