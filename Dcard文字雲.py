#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:19:39 2021

@author: andyhsu
"""

import pandas as pd
import jieba
import re
import emoji
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
jieba.set_dictionary(r'/Users/andyhsu/Desktop/python premium/NLP/dict.txt')
jieba.load_userdict(r'Dcard.txt')

def read_data(data):
    df = pd.read_csv(data,encoding = 'utf-8-sig')
    return df

def combine(df,col,new_col):#參數是dataframe及list和新的col_name
    df[new_col] = ''
    for x in col:
        print(x)
        try:
            df[x] = df[x].astype('str')
            df[new_col] = df[new_col] + ' '+ df[x]
        except:
            print('有誤')
    return df

def clean_space(df,col):
    df[col] = df[col].apply(lambda x:x.strip())
    return df

def remove_emoji(sen):
    new_sen = emoji.demojize(sen)#先將emoji轉成編碼
    new_sen = re.sub(':\S+?:','',new_sen)#在使用RE去除
    return new_sen
    
def remove_emoji_df(df,col):
    df[col] = df[col].apply(lambda x:remove_emoji(x))
    return df

def cut_list(df,col):#參數是dataframe及col_name，return list
    seg_list = []
    for x in df[col].tolist():
        tem_list = jieba.lcut(x)
        seg_list.extend(tem_list)
    return seg_list

def remove_stop_word(seg_list,file):#參數是list及stop_word.text
    remove_list = []
    with open(file,'r') as f:
        stop_words = [sw.rstrip() for sw in f.readlines()]
        for word in seg_list:
            if word not in stop_words:
                remove_list.append(word)
    return remove_list

def count_segment_freq(seg_list):#輸入為list
    seg_df = pd.DataFrame(seg_list,columns=['seg'])
    seg_df['count']=1
    sef_freq = seg_df.groupby('seg')['count'].sum().sort_values(ascending=False)
    sef_freq = pd.DataFrame(sef_freq)
    return sef_freq

dcard_df = read_data('Dcard文章資料.csv')
combine(dcard_df,['Title','內文'],'all_content')
clean_space(dcard_df,'all_content')
remove_emoji_df(dcard_df,'all_content')
seg_list = cut_list(dcard_df,'all_content')
seg_list_no_stop = remove_stop_word(seg_list,'/Users/andyhsu/Desktop/python premium/NLP/停用詞.txt')
print(seg_list_no_stop)
count_segment_freq(seg_list_no_stop)



from PIL import Image
font_path = 'Hiragino Sans GB.ttc'
final_seg = (' ').join(seg_list_no_stop)
import imageio
import numpy as np
mask = np.array(Image.open('pic.jpg'))
wc = WordCloud(background_color = 'black',font_path = font_path)
wc.generate(final_seg)
import matplotlib.pyplot as plt
plt.figure(figsize=(50,50))
plt.imshow(wc)
plt.axis("off")
plt.show()



        