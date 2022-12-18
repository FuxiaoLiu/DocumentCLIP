#!/usr/bin/env python
# -*- coding:utf-8 -*-
#


import json
import multiprocessing as mp
import os
from pathlib import Path
import sys


p = Path(sys.argv[1])

files = list(p.glob("*/doc.json"))


def get_stats(f):
    with open(f) as fi:
        d = json.load(fi)
    return len(d["sections"]), len(d["images"])


with mp.Pool(8) as p:
    res = p.map(get_stats, files)

sec_num_list, img_num_list = zip(*res)
print(f"{len(files):,} articles, {sum(sec_num_list):,} sections, {sum(img_num_list):,} images")
