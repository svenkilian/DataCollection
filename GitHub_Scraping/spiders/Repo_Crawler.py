# -*- coding: utf-8 -*-


import scrapy
from scrapy.crawler import CrawlerProcess
from GitHub_Scraping.items import GithubScrapingItem
import pandas as pd
import json


class RepoCrawlerSpider(scrapy.Spider):
    """
    This class implements a repository crawler which, when given a list of GitHub Repository links,
    crawls through the repositories and creates a json file including the relevant Repository information
    """

    name = 'Repo_Crawler'
    allowed_domains = ['github.com']

    # Retrieve repository list from json file and filter for content
    data = pd.read_json('C:/Users/svenk/Google Drive/[04] Stuff/Query Results/repositories.json', lines=True)
    cnn_data = data[data['repo_name'].str.contains('cnn')]

    # Initialize and populate list from Pandas DataFrame
    repo_list = []
    for item in cnn_data['repo_name']:
        repo_list.append('https://github.com/' + item)

    # Set start urls
    start_urls = repo_list

    def parse(self, response):
        print('Processing: ' + response.url)

        # Crawl repository elements from page
        repo_names = response.xpath("//strong[@itemprop='name']/a/text()").extract()
        repo_desc = response.xpath("//span[@itemprop='about']/text()").extract()
        repo_owner = response.xpath("//span[@itemprop='author']/a/text()").extract()
        repo_ref = [None]
        repo_dcount = [None]
        repo_last_mod = response.xpath("//span[@itemprop='dateModified']/relative-time/@datetime").extract()

        repo_watch = response.xpath(
            "//ul[@class='pagehead-actions']//a[@class='social-count' and contains(@aria-label, 'watching')]/text()").extract()
        repo_stars = response.xpath(
            "//ul[@class='pagehead-actions']//a[@class='social-count js-social-count' and contains(@aria-label, 'starred')]/text()").extract()
        repo_forks = response.xpath(
            "//ul[@class='pagehead-actions']//a[@class='social-count' and contains(@aria-label, 'forked')]/text()").extract()

        # Consolidate data into list
        row_data = zip(repo_names, repo_desc, repo_owner, repo_ref, repo_dcount, repo_last_mod, repo_watch,
                       repo_stars, repo_forks)

        # Populate scraping item with data
        for repo in row_data:
            item = GithubScrapingItem()
            item['page'] = response.url
            item['repo_name'] = repo[1].strip()
            item['repo_link'] = 'https://github.com/' + repo[2].strip() + '/' + repo[0]
            item['repo_owner'] = 'https://github.com/' + repo[2]
            item['repo_ref'] = repo[3]
            item['repo_dcount'] = repo[4]
            item['repo_last_mod'] = repo[5]
            item['repo_watch'] = int(repo[6].strip())
            item['repo_stars'] = int(repo[7].strip())
            item['repo_forks'] = int(repo[8].strip())

            yield item


if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(RepoCrawlerSpider)
    process.start()

    # scraped_data = [json.loads(line) for line in
    #                 open('C:/Users/svenk/PycharmProjects/GitHub_Scraping/data/data.json', 'r', encoding='utf8')]
    # scraped_data_df = pd.DataFrame(scraped_data)
    # scraped_data_df.to_csv('C:/Users/svenk/PycharmProjects/GitHub_Scraping/data/DataLog.csv', sep=',', header=True, index=0)
