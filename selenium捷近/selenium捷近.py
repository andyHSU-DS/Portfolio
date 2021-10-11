#!/usr/bin/env python
# coding: utf-8

# In[2]:


from selenium import webdriver
from time import sleep
import pandas as pd
from pandas import DataFrame
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome('/usr/local/bin/chromedriver')


dfall = pd.read_csv(r'/Users/andyhsu/Desktop/Charge Spot專案/ChargeSPOT_資料集/商家資料(mrt)/有mrt/沒抓到mrt.csv')
df100=dfall.iloc[:100,:]
IDlist = df100['地址'].tolist()
station = []
walk = []

for i in range(0, len(IDlist)):
    address = "https://mrtexit.com/?dest=" + IDlist[i]
    driver.get(address)
    sleep(3)
    sta = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div[2]/div[2]/p[1]').text
    sta = sta + "站"
    station.append(sta)
    wal = driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div[2]/div[2]/p[4]').text
    walk.append(wal)

newdf = DataFrame(walk)
newdf['station'] = station

newdf.to_csv(r'/Users/andyhsu/Desktop/Charge Spot專案/ChargeSPOT_資料集/商家資料(mrt)/有mrt/沒抓到mrt(selenium結果).csv',encoding='utf-8-sig',header=True,index=False)


# In[ ]:


df.iloc[:100,:]


# In[ ]:


df.iloc[101:201,:]


# In[ ]:




