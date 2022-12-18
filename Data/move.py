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

