#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-09-01 14:41:49
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-01 14:58:46
'''


from bs4 import BeautifulSoup
import requests as req
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from zhon.hanzi import punctuation
punct = punctuation + string.punctuation
punct += "―"


def eqcut(text, star=1, end=3):
    words = []
    for k in range(star, end+1):
        for ndx in range(0, len(text)-k+1, 1):
            word = text[ndx:ndx+k]
            words.append(word)
    return words


def getwdata(words):
    # count the words
    wcount = {}
    for word in words:
        if(word in wcount):
            wcount[word] += 1
        else:
            wcount[word] = 1

    # set the output words contents
    wdata = pd.DataFrame.from_dict(wcount, orient='index', columns=["freq"])\
        .sort_values(["freq"], ascending=False)\
        .reset_index()\
        .rename(columns={"index": "word"})

    return wdata


def wcloudplot(wdata, nout=100):
    text_cut = '/'.join(wdata.loc[0:nout, "word"].tolist())
    wordcloud = WordCloud(background_color='white',
                          font_path='Microsoft Yahei.ttf',
                          # mask = imread("pic.png"),
                          width=2500, height=2000, margin=2
                          ).generate(text_cut)

    # show picture
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def readurl(url):
    rep = req.get(url)
    soup = BeautifulSoup(rep.text, 'html.parser')
    text = ""
    for tag in soup.find_all('p'):
        text += tag.text
    return text


def clean(text):
    return re.sub(r"[%s\sA-z0-9]+" % punct, "", text)
