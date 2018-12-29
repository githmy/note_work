# -*- coding: utf-8 -*-
from ghost import Ghost
import re


class Cookieutil:
    def __init__(self, url):
        gh = Ghost(download_images=False, display=False)
        gh.open(url)
        gh.open(url)
        gh.save_cookies("cookie.txt")
        gh.exit()

    def getCookie(self):
        cookie = ''
        with open("cookie.txt") as f:
            temp = f.readlines()
            for index in temp:
                print index
                cookie += self.parse_oneline(index).replace('\"', '')
        return cookie[:-1]

    def parse_oneline(self, src):
        oneline = ''
        if re.search("Set-Cookie", src):
            oneline = src.split(';')[0].split(':')[-1].strip() + ';'
        return oneline


if __name__ == "__main__":
    pass
