# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class FetcherItem(scrapy.Item):
    # define the fields for your item here like:
    id = scrapy.Field()
    na = scrapy.Field()
    op = scrapy.Field()
    cl = scrapy.Field()
    hi = scrapy.Field()
    lo = scrapy.Field()
    av = scrapy.Field()
    pass
