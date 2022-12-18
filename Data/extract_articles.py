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
Extract image and category information from a Wikiepdia dump
"""

# Built-in modules
import argparse
import bz2
from collections import defaultdict
import json
import multiprocessing as mp
import os
import re
from xml.etree import ElementTree as ET

# Third-party modules
from gensim.corpora.wikicorpus import get_namespace, filter_wiki, RE_P14, RE_P15
from gensim.scripts.segment_wiki import extract_page_xmls, segment_all_articles
from gensim import utils


def main():
    args = parse_opt()

    os.makedirs(args.out_dir, exist_ok=True)

    saved, skipped, processed = 0, 0, 0
    excluded_arts = set()
    arts_sec = segment_all_articles(args.wiki_dump, include_interlinks=True)
    arts_img_cat = parse_all_articles(args.wiki_dump)
    for art in arts_sec:
        processed += 1
        title, sections, interlinks = art
        sec_headers, sec_contents = split_htag(sections)
        is_none = False
        while 1:
            title_2, doc_id, categories, images, sec_heads_2, raw_text, tables = next(arts_img_cat)
            if title_2 == title:
                break
            elif title_2 is None:
                is_none = True
                break
            else:
                skipped += 1
        if is_none:
            skipped += 1
            continue

        if sec_heads_2 != sec_headers:
            skipped += 1
            print(f"Skipped due to different header: {title}")
            continue

        if any(map(lambda c: is_non_target_category(c), categories)):
            excluded_arts.add(title)
            skipped += 1
            continue

        sec_contents = remove_caption(sec_contents, images)
        sec_contents = remove_table(sec_contents, tables)

        xml_obj = formatter(
            doc_id, title, sec_headers, sec_contents, images
        )

        art_dir = os.path.join(args.out_dir, title.replace("/", "_"))
        if os.path.exists(art_dir):
            skipped += 1
            continue
        os.mkdir(art_dir)

        art_file = os.path.join(art_dir, "doc.xml")
        ET.ElementTree(xml_obj).write(art_file)
        il_file = os.path.join(art_dir, "interlinks.json")
        with open(il_file, "w", encoding="utf-8") as fo:
            json.dump({v: k for k, v in interlinks.items()},
                      fo, ensure_ascii=False)
        rt_file = os.path.join(art_dir, "raw_text.txt")
        with open(rt_file, "w", encoding="utf-8") as fo:
            fo.write(raw_text)
        ct_file = os.path.join(art_dir, "category.txt")
        with open(ct_file, "w", encoding="utf-8") as fo:
            fo.write("\n".join(categories) + "\n")
        info = get_info(xml_obj)
        mt_file = os.path.join(art_dir, "meta.json")
        with open(mt_file, "w", encoding="utf-8") as fo:
            json.dump(info, fo)

        saved += 1

    excluded_file = os.path.join(args.out_dir, "excluded_articles.txt")
    with open(excluded_file, "w", encoding="utf-8") as fo:
        fo.write("\n".join(sorted(excluded_arts)))
    print(f"Processed {processed} articles, {saved} saved, {skipped} skipped.")
    return


def parse_opt():
    """Parse command line option.

    Returns
    -------
    args : argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("wiki_dump", help="Wikipedia's dump file.", type=str)
    parser.add_argument("-o", "--out_dir", help="output directory.",
                        type=str, default="articles")

    args = parser.parse_args()
    return args


def parse_all_articles(xml_file):
    parallel_n = max(1, mp.cpu_count() - 1)
    fi = bz2.BZ2File(xml_file)
    p = mp.Pool(parallel_n)
    for g in utils.chunkize(extract_page_xmls(fi),
                            chunksize=10*parallel_n, maxsize=1):
        for art in p.imap(parse_xml, g):
            yield art
    p.terminate()
    fi.close()


def parse_xml(page_xml):
    """Parse XML for an entire Wikipedia article.

    Parameters
    ----------
    page_xml : str
        XML markup string.

    Returns
    -------
    (str, list of str, list of (int, str))
        article title, categories, and image info.
    """
    elem = ET.fromstring(page_xml)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}

    id_path = "./{%(ns)s}id" % ns_mapping
    doc_id = elem.find(id_path).text
    title_path = "./{%(ns)s}title" % ns_mapping
    title = elem.find(title_path).text
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    text = elem.find(text_path).text
    if not isinstance(text, (bytes, str)):
        return [None]*7
    categories = get_category_info(text)
    images, sec_heads = get_image_info(text)
    table_texts = get_table_info(text)

    return title, doc_id, categories, images, sec_heads, text, table_texts


CAT_PTN = re.compile(r"^\[\[Category:(.+)\]\]$")
def get_category_info(text):
    """Returns category information for given article.

    Parameters
    ----------
    text : str
        Raw text of an entire Wikipedia article.

    Returns
    -------
    list of str
        Wikipedia's categories.
    """
    return [re.sub(CAT_PTN, r"\1", c) for c in re.findall(RE_P14, text)]


H_HEADER_RAW = re.compile(r"(?<=[^=])={2,6}[^=].*?[^=]={2,6}(?=[^=])")
H_HEADER_RAW_G = re.compile(r"(?<=[^=])={2,6}([^=].*?[^=])={2,6}(?=[^=])")
def get_image_info(text):
    """Return image information along with section IDs for given article.

    Parameters
    ----------
    text : str
        Raw text of an entire Wikipedia article.

    Returns
    -------
    list of (int, str)
        List of (Section ID, Image info), where Image info contains
         filename and caption.
    """
    section_contents = re.split(H_HEADER_RAW, text)
    imgs = []
    for i, sec_text in enumerate(section_contents):
        for match in re.finditer(RE_P15, sec_text):
            s, xe = match.regs[1]  # start and end positions of "File:|Image" in section
            e, _ = get_image_tag_end(sec_text, xe+1)
            imgs.append((i, *split_itag(sec_text[s-2:e+2])))
    section_headers = ["Introduction"] + [filter_wiki(h).strip() for h in re.findall(H_HEADER_RAW_G, text)]
    assert len(section_contents) == len(section_headers)
    return imgs, section_headers


def get_image_tag_end(text, start=0):
    opened = 1
    pos = start
    while opened > 0 and pos <= len(text):
        ch = text[pos:pos+2]
        if ch == "]]":
            opened -= 1
            pos += 2
        elif ch == "[[":
            opened += 1
            pos += 2
        else:
            pos += 1
    pos -= 2
    return pos, text[start:pos]


def get_table_info(text):
    tables = []
    section_contents = re.split(H_HEADER_RAW, text)
    for i, sec_text in enumerate(section_contents):
        if "wikitable" in sec_text:
            table_seg = []
            in_table = False
            for l in sec_text.split("\n"):
                if "wikitable" in l:
                    in_table = True
                if in_table and "|}" in l:
                    table_raw = "\n".join(table_seg)
                    tables.append((i, filter_wiki(table_raw)))
                    in_table = False
                    table_seg = []
                if in_table:
                    table_seg.append(l)
    return tables

# Resplit with <h2> tags since the original gensim regex fails to split with <h2> tags
TOP_HEADER = re.compile(r"={2,6}[^=].*[^=]={2,6}\n")
TOP_HEADER_G = re.compile(r"={2,6}([^=].*[^=])={2,6}\n")
def split_htag(sections):
    """Split section texts with <h> tags.

    Parameters
    ----------
    sections : (list of str, list of str)
        First list consists of headers while the second has section texts.

    Returns
    -------
    (list of str, list of str)
    """
    headers, contents = [], []
    for h, s in sections:
        headers.append(filter_wiki(h))
        s = "\n" + s  # to split the first tag
        h_contents = re.split(TOP_HEADER, s)
        h_headers = re.findall(TOP_HEADER_G, s)
        h_headers = [filter_wiki(h).strip() for h in h_headers]
        assert len(h_contents) == len(h_headers) + 1
        headers += h_headers
        contents += h_contents
    return headers, contents


def remove_caption(secs, imgs):
    """Remove caption from section text.

    Parameters
    ----------
    secs : list of str
        List of section texts.
    imgs : List of (int, str)
        List of pairs of section ids and captions.

    Returns
    -------
    list of str
    """
    for sec_i, _, cap in imgs:
        secs[sec_i] = secs[sec_i].replace(cap, "")
    return secs


def remove_table(secs, tbls):
    """Remove table from section text.

    Parameters
    ----------
    secs : list of str
        List of section texts.
    tbls : List of str
        List of pairs of section id and table text.

    Returns
    -------
    list of str
    """
    for sec_i, tbl in tbls:
        secs[sec_i] = secs[sec_i].replace(tbl, "").strip()
    return secs


URL_BASE = "https://en.wikipedia.org/wiki?curid="
RE_SENT = re.compile(r'(?<=[.?!])[ \n]+(?=[A-Z])')
def formatter(doc_id, title, header, secs, images):
    header[0] = title

    sid2img = defaultdict(list)
    for sid, name, cap in images:
        sid2img[sid].append((name, cap))
    img_counter = 0

    obj = ET.Element("doc")
    obj.set("id", doc_id)
    obj.set("url", URL_BASE + str(doc_id))
    obj.set("title", title)
    obj.text = "\n"
    for i, (h, s) in enumerate(zip(header, secs)):
        sec = ET.SubElement(obj, "section")
        sec.set("id", str(i))
        sec.set("title", h)
        sec.text = "\n" + "\n".join(re.split(RE_SENT, s.strip())) + "\n"
        if i in sid2img:
            for name, cap in sid2img[i]:
                img = ET.SubElement(sec, "image")
                img.set("id", str(img_counter))
                img.set("name", name)
                img.text = "\n" + cap + "\n"
                img_counter += 1

    return obj


IMG_ATTR = re.compile(r"^thumb(nail)?|frame(d|less)?|boader|right|left|center|none|baseline|middle|sub|super|text-top|text-bottom|top|bottom|upright(=[0-9.]+)?|\d+?x?\d+px|(link|alt|page|lang)=.*$")
def split_itag(itag):
    # For the detail of Image Syntax, see https://en.wikipedia.org/wiki/Wikipedia:Extended_image_syntax
    # Brief syntax: [[File:Name|Type|Border|Location|Alignment|Size|link=Link|alt=Alt|page=Page|lang=Langtag|Caption]]
    if len(itag.split("|")) == 1:
        filename = filter_wiki(itag)
        caption = ""
    else:
        filename = itag.split("|")[0][2:]
        caption = filter_wiki("|".join([e for e in itag[2:-2].split("|")[1:]
                                        if len(re.sub(IMG_ATTR, "", e.strip())) != 0]))
        caption = caption.replace("\n", " ")
    return filename, caption


RE_TC = re.compile(r"(\d+ birth|[Pp]eople|[Ss]port|[Ss]eason|[Mm]atch)")
def is_non_target_category(category):
    m = re.search(RE_TC, category.lower())
    if m is None:
        return False
    else:
        return True


def get_info(doc):
    sections = doc.findall("section")
    images = [i for s in sections for i in s.findall("image")]

    article_name = doc.get("title")
    text_length = sum([len(s.text) for s in sections])
    section_num = len(sections)
    image_num = len(images)
    image_filenames = [i.get("name") for i in images]

    info = {
        "article_name": article_name,
        "text_length": text_length,
        "section_num": section_num,
        "image_num": image_num,
        "image_filenames": image_filenames
    }
    return info


if __name__ == "__main__":
    main()
