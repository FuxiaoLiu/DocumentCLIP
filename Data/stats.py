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
