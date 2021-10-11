#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:30:03 2021

@author: andyhsu
"""
import pandas as pd
import requests
import time
import random
import re
validips = []
free_proxy = 'https://free-proxy-list.net/'
free_proxy_list = requests.get(free_proxy).text
ips = re.findall(r'\d+\.\d+\.\d+\.\d+:\d+',free_proxy_list)
print(ips)
for ip in ips:
    try:
        res = requests.get('https://api.ipify.org?format=json',proxies ={'htttp':ip,'https':ip},timeout = 5)
        res.json()
        validips.append(ip)
    except:
        print('FAIL',ip)
    
    
    
    
print(validips)
###----------Dcard內容--------###
titles = []
excerpts = []
comment_num = []
like_count =[]

chapter_title = []
comment_content = []
comment_like = []
    
headers = {
    'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'
}
website = 'https://www.dcard.tw/service/api/v2/forums/nccu/posts?popular=true&limit=100'

round_ = 0

while round_<3:
    print('-'*50)
    print(round_)
    print(website)
    chapters = requests.get(website,headers = headers).json()
    for chapter in chapters:
        time.sleep(random.randint(2,6))
        print('標題:',chapter['title'])
        print('內文:',chapter['excerpt'])
        print('留言數:',chapter['commentCount'])
        print('喜歡數:',chapter['likeCount'])
        
        titles.append(chapter['title'])
        excerpts.append(chapter['excerpt'])
        comment_num.append(chapter['commentCount'])
        like_count.append(chapter['likeCount'])
        
        id_ = chapter['id']
        if chapter['commentCount'] != 0: #沒留言不用看
            comment_id = 'https://www.dcard.tw/service/api/v2/posts/'+str(id_)+'/comments?'
            try:
                comment_info = requests.get(comment_id,headers = headers).json()
                for com in comment_info:
                    try:
                        print(chapter['title'])
                        print('留言內容:',com['content'])
                        print('留言讚數:',com['likeCount'])
                        chapter_title.append(chapter['title'])
                        comment_content.append(com['content'])
                        comment_like.append(com['likeCCount'])
                    except:
                        print('留言已刪除')
            except:
                print('讀取json有問題：',comment_id)
    website = 'https://www.dcard.tw/service/api/v2/forums/nccu/posts?limit=100'
    website = website+'&before='+str(id_)
    round_ += 1
    
    
dcard = pd.DataFrame({'Title':titles,'內文':excerpts,'留言數':comment_num,'按讚數':like_count})
dcard.to_csv('Dcard留言資料.csv',header=True,index=False,encoding='utf-8-sig')
    
    
    

