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
"""
Remove personal information based on human-related article names.
"""

# Built-in modules
import argparse
from collections import defaultdict
import html
import json
import os
import re
from xml.etree import ElementTree as ET

from extract_articles import get_info, RE_SENT

XML_FILE = "doc.xml"
META_JSON = "meta.json"


def main():
    args = parse_opt()

    article_dirs = os.listdir(args.art_dir)
    if "excluded_articles.txt" in article_dirs:
        ea_file = os.path.join(args.art_dir, "excluded_articles.txt")
        with open(ea_file) as fi:
            pi_set = set([l.strip() for l in fi])
        pi = True
    else:
        pi_set = set([os.path.basename(d) for d in article_dirs
                      if os.path.isdir(d)])
        pi = False
    for a in article_dirs:
        art_dir = os.path.join(args.art_dir, a)
        if not os.path.isdir(art_dir):
            continue
        in_file = os.path.join(art_dir, XML_FILE)
        if os.path.exists(in_file.replace(XML_FILE, "doc.org.xml")):
            continue
        anonymized_xml = remove_pi(in_file, pi_set, pi)
        if anonymized_xml is None:
            continue
        os.rename(in_file, in_file.replace(XML_FILE, "doc.org.xml"))
        ET.ElementTree(anonymized_xml).write(in_file)

        info = get_info(anonymized_xml)
        meta_file = os.path.join(art_dir, META_JSON)
        os.rename(meta_file, meta_file.replace(META_JSON, "meta.org.json"))
        with open(meta_file, "w") as fo:
            json.dump(info, fo)

    return


def parse_opt():
    """Parse command line option.

    Args: No args

    Returns
    -------
    args : argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-d", "--art_dir", help="article directory.",
                        type=str, default="./articles")

    args = parser.parse_args()
    return args


def remove_pi(in_file, pi_set, pi):

    def anonymize(t):
        ## unescape text
        une_txt = html.unescape(t)
        # split into sentences
        sents = [sent for nl in une_txt.split("\n")
                 for sent in re.split(RE_SENT, nl)]
        # remove sentences which have person names
        left_sents = [sent for sent in sents
                      if len(re.findall(psn_re, sent)) == 0]
        ano_txt = "\n".join(left_sents).strip()
        return ano_txt

    # Get all text that has inter links
    art_dir = os.path.dirname(in_file)
    link_file = os.path.join(art_dir, "interlinks.json")
    with open(link_file) as fi:
        links = json.load(fi)
    # Get person names found in the article
    if pi:
        found_persons = set(links.values()) & pi_set
    else:
        found_persons = set(links.values()) - pi_set
    fp_text = set([k for k, v in links.items() if v in found_persons])
    try:
        psn_re = re.compile(fr'{"|".join(fp_text)}')
    except:
        return None

    # Load xml file
    xml = ET.parse(in_file).getroot()
    if len(fp_text) == 0:
        return xml
    # Remove all expressions found in the sections
    sections = xml.findall("section")
    removed = defaultdict(list)
    for s in sections:
        old_title = s.get("title")
        new_title = anonymize(old_title)
        if old_title != new_title:
            removed[old_title].append(("titile", old_title, new_title))
            s.set("title", new_title)
        alltxt = " ".join([t for t in s.itertext()])
        for i in s.findall("image"):
            alltxt.replace(i.text, "")
            old = i.text
            i.text = anonymize(i.text)
            if len(i.text) != len(old):
                removed[old_title].append(("image", old, i.text))
                s.remove(i)
        s.text = anonymize(alltxt)
        if len(s.text) != alltxt:
            removed[old_title].append(("section", alltxt, s.text))
        if len(s.text) == 0:
            xml.remove(s)
    rm_file = os.path.join(art_dir, "removed.json")
    with open(rm_file, "w", encoding="utf-8") as fo:
        json.dump(removed, fo, indent=2, ensure_ascii=False)
    return xml


if __name__ == "__main__":
    main()
