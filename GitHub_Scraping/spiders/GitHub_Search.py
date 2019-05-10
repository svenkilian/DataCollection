# -*- coding: utf-8 -*-

import scrapy
from scrapy.crawler import CrawlerProcess
from GitHub_Scraping.items import GitHubRepositoryItem


class GitHub_Search_Spider(scrapy.Spider):
    name = 'GitHub_Search'
    allowed_domains = ['github.com']

    start_urls = ['https://github.com/search?l=Python&p=' + str(p) + '&q=keras&type=Repositories' for p in range(1, 20)]

    def parse(self, response):
        print('Processing: ' + response.url)

        repo_names = response.xpath("//h3/a/@href").extract()

        for repo in repo_names:
            item = GitHubRepositoryItem()
            item['repo_name'] = repo[1:]
            yield item


if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(GitHub_Search_Spider)
    process.start()
