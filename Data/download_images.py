#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://en.wikipedia.org/wiki/"
# Do NOT set less than 1 to the value of `SLEEP` because heavy access requests cause high pressure on the Wikipedia's server.
# For more detail, see https://en.wikipedia.org/wiki/Wikipedia:Database_download#Please_do_not_use_a_web_crawler
SLEEP = 3


def load_json(dir):
    with open(os.path.join(dir, "meta.json"), "r") as f:
        return json.load(f)


def post_request(url):
    try:
        time.sleep(SLEEP)
        req = requests.get(url, stream=True, timeout=5)
        if req.status_code == 404 and "' " in url:
            return post_request(url.replace("' ", "'"))
        if req.status_code != 200:
            raise Exception("Bad request status : {} ({})".format(req.status_code, url))
        return req
    except TimeoutError:
        raise Exception("Timeout")


def get_soup(url):
    r = post_request(url)
    return BeautifulSoup(r.text, 'lxml')


def download(image_name, has_all_img):
    if has_all_img:
        return None, None, None
    url = os.path.join(BASE_URL, image_name)
    soup = get_soup(url)
    image_license = get_license_info(soup)
    # Discard images if the license does not grant to use.
    # Well-known licenses: https://commons.wikimedia.org/wiki/Commons:Licensing#Well-known_licenses
    if not is_ok(image_license):
        return None, soup, image_license
    image_tag = soup.findAll("div", class_="fullImageLink")
    if len(image_tag) != 1:
        raise Exception("The marker tag fullImageLink was not 1.")
    image_tag = image_tag[0]
    image_url = "https:" + image_tag.find("a").get("href")
    print("try ... ", image_url)
    req = post_request(image_url)
    return req.content, soup, image_license


def get_license_info(soup):
    return "\n".join([l.text.strip() for l in soup.findAll("table", class_="layouttemplate") or soup.findAll("table", class_="licensetpl") if l is not None])

lps = [re.compile(fr".*?{l}.*?\n")
       for l in ["Creative Commons",  # OK
                 "GNU Free Documentation License",  # OK
                 "allows anyone to use it for any purpose",  # OK
                 "[Pp]ublic [Dd]omain",  # NG
                 "A normal copyright tag is still required.",  # NG
                 "[Ff]air [Uu]se",  # NG
                 ]]
def is_ok(license):
    licenses = list(map(lambda x: re.findall(x, license), lps))
    count_list = [1 if len(ls) != 0 else 0 for ls in licenses]
    count_ok = sum(count_list[:2])
    count_ng = sum(count_list[3:])
    if count_ok > 0 and count_ng == 0:
        return True
    if count_ng > 0:
        return False
    if count_ok + count_ng == 0:
        return False
    return False


def has_all(dir, img_names):
    for i, name in enumerate(img_names):
        filename = os.path.join(dir, "{}-{}".format(i, name))
        if not os.path.exists(filename):
            return False
    return True


def save(dir, image_names, image_binaries, soups, licenses):
    for i, (name, image, soup, license) in enumerate(zip(image_names, image_binaries, soups, licenses)):
        if image is not None:
            with open(os.path.join(dir, "{}-{}".format(i, name)), "wb") as f:
                f.write(image)
        if soup is not None:
            with open(os.path.join(dir, "{}-{}.soup".format(i, name)), "w") as f:
                f.write(str(soup))
        if license is not None:
            with open(os.path.join(dir, "{}-{}.license".format(i, name)), "w") as f:
                f.write(license + "\n")


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser("Download wikipedia images", add_help=True)

    args.add_argument("-d", help="Directory including article subdirectories", required=True)
    args.add_argument("-l", help="Filtered article list (when you wanna do download only for specific articles)", default=None)
    args.add_argument("-o", help="List of articles which successfully downloding all images", default="./success_image_download_articles_list.txt")
    args.add_argument("-sleep", help="Request Delay", default=SLEEP)

    opt = args.parse_args()

    SLEEP = float(opt.sleep)

    if opt.l is None:
        articles = os.listdir(opt.d)
    else:
        with open(opt.l, "r") as f:
            articles = [l.strip() for l in  f]

    success_directories = []

    for a in articles:
        d = os.path.join(opt.d, a)
        metainfo = load_json(d)

        nimg = metainfo["image_num"]
        img_names = metainfo["image_filenames"]

        try:
            has_all_img = has_all(d, img_names)
            if has_all_img:
                print("{} has already images.".format(d))
            img_binaries, soups, licenses = zip(*[download(i, has_all_img) for i in img_names])
            save(os.path.join(opt.d, a), img_names, img_binaries, soups, licenses)
            print("DONE: {}".format(a))
            success_directories.append(a)
        except Exception as e:
            sys.stderr.write("\nFailed to download (some) images in the article {}\n".format(a))
            sys.stderr.write(str(e) + "\n")

    with open(opt.o, "w") as f:
        for l in success_directories:
            f.write(l + "\n")


