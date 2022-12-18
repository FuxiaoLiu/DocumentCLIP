#!/usr/bin/env python
# -*- coding:utf-8 -*-


import json
import os
import sys


def load_json(dir):
    with open(os.path.join(dir, "meta.json"), "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser("Filter articles by your conditions", add_help=True)

    args.add_argument("-d", help="Directory including article subdirectories", required=True)
    args.add_argument("-maxlen", help="Minimum number of length", default=sys.maxsize)
    args.add_argument("-maxsec", help="Minimum number of sections", default=sys.maxsize)
    args.add_argument("-maximg", help="Minimum number of images", default=sys.maxsize)
    args.add_argument("-minlen", help="Minimum number of length", default=-1)
    args.add_argument("-minsec", help="Minimum number of sections", default=10)
    args.add_argument("-minimg", help="Minimum number of images", default=10)

    args.add_argument("-o", help="Filtered article list", default="./target_articles.txt")


    opt = args.parse_args()

    article_dirs = os.listdir(opt.d)

    target_articles = []

    def is_ok(metainfo):
        tlen = metainfo["text_length"]
        snum = metainfo["section_num"]
        inum = metainfo["image_num"]

        if tlen < int(opt.minlen) or int(opt.maxlen) < tlen:
            return False

        if snum < int(opt.minsec) or int(opt.maxsec) < snum:
            return False

        if inum < int(opt.minimg) or int(opt.maximg) < inum:
            return False

        return True

    for a in article_dirs:
        art_dir = os.path.join(opt.d, a)
        if not os.path.isdir(art_dir):
            continue
        metainfo = load_json(art_dir)
        if is_ok(metainfo):
            target_articles.append(a)

    with open(opt.o, "w") as f:
        for a in target_articles:
            f.write(a + "\n")
