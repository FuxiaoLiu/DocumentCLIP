#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import shutil
import sys

files = []
with open(sys.argv[1], "r") as f:
    for l in f:
        files.append(l.strip())

trash_dir = sys.argv[2]

for f in files:
    if os.path.exists(f):
        shutil.move(f, trash_dir)

