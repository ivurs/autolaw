import fitz
import streamlit as st
import fitz
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd
import numpy as np
from operator import itemgetter
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
import pickle

def fonts(doc, granularity=False):
    """Extracts fonts and their usage in PDF documents.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool

    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage
    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)
    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")
    return font_counts, styles


def font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.

    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict

    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag
    font_sizes = []
    for (font_size, count) in font_counts:
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)
    return size_tag


def get_font_style_size_tag(doc):
    font_counts, styles = fonts(doc, granularity=False)
    size_tag = font_tags(font_counts, styles)
    return size_tag

def get_clean_element(element):
    return [i.strip() for i in element.split("|")]

def get_related_pages(doc):
    related_pages = []
    for i in range(doc.page_count):
            page = doc.load_page(i)
            page_to_text = page.get_text("text").replace("\n", " ").lower()
            if 'breach of copyright may result in legal action' not in page_to_text:
                related_pages.append(i)
    return related_pages

def headers_para(page, size_tag, exclude_size_tag=True):
    """Scrapes headers & paragraphs from PDF and return texts with element tags.

    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param size_tag: textual element tags for each size
    :type size_tag: dict

    :rtype: list
    :return: texts with pre-prended element tags
    """
    header_para = []  # list with headers and paragraphs
    first = True  # boolean operator for first header
    previous_s = {}  # previous span
    #block_bound_boxes = []
    #for p in pages:
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:  # iterate through the text blocks
        if b['type'] == 0:  # this block contains text
            #block_bound_boxes.append(b['bbox'])
            # REMEMBER: multiple fonts and sizes are possible IN one block
            block_string = ""  # text found in block
            for l in b["lines"]:  # iterate through the text lines
                for s in l["spans"]:  # iterate through the text spans
                    if s['text'].strip():  # removing whitespaces:
                        if first:
                            previous_s = s
                            first = False
                            if exclude_size_tag:
                                block_string = s['text']
                            else:
                                block_string = size_tag[s['size']] + s['text']
                        else:
                            if s['size'] == previous_s['size']:

                                if block_string and all((c == "|") for c in block_string):
                                    # block_string only contains pipes
                                    if exclude_size_tag:
                                        block_string = s['text']
                                    else:
                                        block_string = size_tag[s['size']] + s['text']
                                if block_string == "":
                                    # new block has started, so append size tag
                                    if exclude_size_tag:
                                        block_string = s['text']
                                    else:
                                        block_string = size_tag[s['size']] + s['text']
                                else:  # in the same block, so concatenate strings
                                    block_string += " " + s['text']
                            else:
                                header_para.append(block_string)
                                if exclude_size_tag:
                                    block_string = s['text']
                                else:
                                    block_string = size_tag[s['size']] + s['text']
                            previous_s = s
                # new block started, indicating with a pipe
                block_string += "|"
            header_para.append(block_string)
    return header_para

def get_all_sentence_by_file(filename,doc):
    size_tag = get_font_style_size_tag(doc)
    all_pages = get_related_pages(doc)
    cur_doc_ctc = []
    for page in all_pages:
        cur_page_content = headers_para(doc.load_page(page), size_tag)
        for cpc in cur_page_content:
            cur_doc_ctc.append([cpc.replace("|",""), filename.split(".pdf")[0], page+1])
    return pd.DataFrame(np.array(cur_doc_ctc), columns= ['sentence', 'contract_name','page'])


def get_topic_dict(files):
    cur_dict = {}
    for fn in files:
        with open(fn, "rb") as f :
            cur_dict.update(pickle.load(f)) 
    return cur_dict