# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from video_scra.log_tool import logger

class VideoScraPipeline(object):
    def process_item(self, item, spider):
        return item

    def make_seed(self,keylist):
        for i1 in keylist:
            print(i1)
