#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os


# In[2]:


file = os.getcwd()


# In[3]:


mel = pd.read_csv(file+'/Melbourne_housing_FULL.csv',parse_dates = ['Date'])


# # 資料初探

# In[4]:


mel.head()


# In[5]:


mel.nunique()


# In[6]:


mel.info()


# In[7]:


na_df = pd.DataFrame(mel.isna().sum()/len(mel),columns = ['NaN percent(%)'])
print(na_df)
na_df['NaN percent(%)'] = na_df['NaN percent(%)'].apply(lambda x:round(x,2))
na_df


# In[8]:


px.bar(na_df,x=na_df.index,y='NaN percent(%)',text='NaN percent(%)',title='個欄位缺失值百分比',
      labels={'index':'欄位名'})


# In[9]:


mel.describe()


# In[10]:


mel_p = mel.dropna(how='any',axis=0)


# In[93]:


mel_p.reset_index(drop=True,inplace=True)


# In[94]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
mel_p['CouncilArea_le'] = labelencoder.fit_transform(mel_p['CouncilArea'])
for x in range(len(mel_p)):
    mel_p.loc[x,'text'] = mel_p.loc[x,'CouncilArea']+':'+str(mel_p.loc[x,'Price'])


# In[95]:


mel_p.head(100)


# In[96]:


fig = go.Figure(data=go.Scattergeo(
        lon = mel_p.head(1000)['Longtitude'],
        lat = mel_p.head(1000)['Lattitude'],
        text = mel_p.head(1000)['text'],
        mode = 'markers',
        marker = {
            'color':mel_p.head(1000)['CouncilArea_le'],
            'size':mel_p.head(1000)['Price']/100000,
                 },
        )
        )

fig.update_layout(
        title = 'Housing Price'
    )
fig.show()


# In[14]:


#因為要用線性回歸模型，去除outlier很重要
mel = mel[(mel['Bedroom2']<=5)&(mel['Bedroom2']>=1)]


# In[15]:


mel.drop(columns = ['Address','Longtitude','Lattitude','BuildingArea','Postcode'],inplace = True)


# In[16]:


mel.info()


# In[17]:


total_agg = mel[mel["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").agg(['sum','mean','std'])


# In[18]:


plt.figure(figsize=(10,20))
plt.subplot(2,1,1)
total_agg['Price']['mean'].plot()
plt.title('mean change group by Date',fontsize=20)
plt.legend('Mean')
plt.subplot(2,1,2)
total_agg['Price']['std'].plot()
plt.title('std change group by Date',fontsize=20)
plt.legend('Std')


# In[19]:


#為了要把suburb轉成數字，依照他們的出售價格平均排序
sub_price_df = pd.DataFrame(mel.groupby(['Suburb','SellerG'])['Price'].mean()).sort_values(by = 'Price')
price_dict = sub_price_df.to_dict()['Price']
sub_price_df.reset_index(inplace=True)
sub_price_df


# In[20]:


mel.reset_index(inplace = True)
mel['Suburb']


# In[21]:


mel['SellerG']


# In[22]:


mel_withoutp = mel[mel['Price'].isnull()]
mel_withoutp.reset_index(inplace=True,drop=True)
mel_withoutp.drop(columns = ['Price'],inplace=True)
mel_withoutp = pd.merge(mel_withoutp,sub_price_df,on=['Suburb','SellerG'],how='left')
mel_withoutp


# In[23]:


melwithp = mel[mel['Price'].isnull()==False]


# In[24]:


mel = pd.concat([melwithp,mel_withoutp[melwithp.columns]],axis=0)
mel.reset_index(inplace=True,drop=True)
mel.drop(columns = ['index'],inplace=True)


# In[25]:


mel


# In[26]:


mel.info()


# In[27]:


mel['Bathroom'].describe()


# In[28]:


room_bed2 = pd.DataFrame(mel.groupby(['Rooms','Bedroom2'])['Bathroom'].value_counts())
room_bed_dict = room_bed2.to_dict()['Bathroom']


# # 用線性模型預測bathrooms數量

# In[29]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[30]:


mel['Rooms & Bedroom2'] = mel['Rooms']+mel['Bedroom2']


# In[31]:


withoutbath=mel[mel['Bathroom'].isnull()]


# In[32]:


withbath = mel[mel['Bathroom'].isnull()==False]


# In[33]:


lr.fit(withbath[['Rooms','Bedroom2']],withbath['Bathroom'])


# In[34]:


pre_bath = lr.predict(withoutbath[['Rooms','Bedroom2']])


# In[35]:


pre_bath


# In[36]:


withoutbath['Bathroom'] = pre_bath


# In[37]:


mel = pd.concat([withbath,withoutbath],axis=0).reset_index(drop = True)
mel['Bathroom'] = mel['Bathroom'].apply(lambda x:round(x,0))
mel


# # 處理停車位

# In[38]:


car_dict = mel.groupby('Type')['Car'].mean().to_dict()
car_dict


# In[39]:


car_dict['h']


# In[40]:


mel_car = mel[mel['Car'].isnull()==False]
mel_nocar = mel[mel['Car'].isnull()]


# In[41]:


mel_nocar['Car'] = mel['Type'].map(car_dict)
mel = pd.concat([mel_car,mel_nocar],axis = 0).reset_index(drop=True)


# In[42]:


mel['Car']=mel['Car'].apply(lambda x:round(x,0))


# In[43]:


y = mel.groupby('CouncilArea')['YearBuilt'].value_counts()
y


# # 處理yearbuilt

# In[44]:


yb_dict = {}
for x in set(y.index.get_level_values(0)):
    print(x)#councilarea
    t = y[y.index.get_level_values(0)==x]
   #找最多的那個
    g = t.argmax()
    print(t.index.get_level_values(1)[g])
    year = t.index.get_level_values(1)[g]
    yb_dict[x] = year


# In[45]:


yb_dict


# In[46]:


mel_withyb = mel[mel['YearBuilt'].isnull()==False]
mel_withoutyb = mel[mel['YearBuilt'].isnull()]


# In[47]:


mel_withoutyb['YearBuilt'] = mel_withoutyb['CouncilArea'].map(yb_dict)


# In[48]:


mel = pd.concat([mel_withyb,mel_withoutyb],axis=0).reset_index(drop=True)
mel


# In[49]:


mel['sell_year'] = mel['Date'].apply(lambda x:x.year)


# In[50]:


mel['Landsize'] = mel['Landsize'].fillna(mel['Landsize'].mean())


# In[51]:


sns.distplot(np.log(mel['Price']))


# In[52]:


mel['house_age'] = mel['sell_year']-mel['YearBuilt']
def fillyear(x):
    if x<0:
        return 0
    else:
        return x


# In[53]:


mel['house_age'] = mel['house_age'].apply(fillyear)


# In[54]:


mel


# # 轉換成數值資料，做get_dummies

# In[55]:


mel.drop(columns = ['Date','Rooms & Bedroom2','sell_year'],inplace=True)


# In[56]:


mel.nunique()


# In[57]:


mel_with_p = mel[mel['Price'].isna()==False]
sellerG_sort = pd.DataFrame(mel.groupby('SellerG')['Price'].agg("mean")).reset_index().sort_values(by='Price',ascending=False)
sellerG_sort


# In[58]:


#排序
a = np.arange(len(sellerG_sort))+1
a


# In[59]:


sellerG_sort['rank'] = a


# In[60]:


sellerG_sort


# In[61]:


#sellerG_sort補rank
for x in range(len(sellerG_sort)):
    if pd.isnull(sellerG_sort.loc[x,'Price'])==True:
        sellerG_sort.loc[x,'rank'] = 323 


# In[62]:


sellerG_sort


# In[63]:


mel = pd.merge(mel,sellerG_sort[['SellerG','rank']],on='SellerG',how='left')


# In[64]:


mel = mel.rename(
    columns={'rank':'sellerRank'}
)


# In[65]:


type_oe = pd.get_dummies(mel['Type'],prefix='Type',drop_first = True)
type_oe


# In[66]:


mel = pd.concat([mel,type_oe],axis=1)
mel


# In[67]:


mel.nunique()


# In[68]:


reg_name_oe = pd.get_dummies(mel['Regionname'],prefix = 'reg',drop_first = True)
reg_name_oe


# In[69]:


mel = pd.concat([mel,reg_name_oe],axis=1)
mel


# In[70]:


method_oe = pd.get_dummies(mel['Method'],prefix = 'Method',drop_first = True)
method_oe


# In[71]:


mel = pd.concat([mel,method_oe],axis=1)
mel


# # 跑預測

# In[72]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[73]:


data = mel.dropna(subset=['Price'],axis=0)
data


# In[74]:


lr = LinearRegression()


# In[75]:


data.columns


# In[76]:


X = data.drop(columns = ['Suburb','Type','Method','SellerG','CouncilArea','YearBuilt','Regionname','Price'])
y = data['Price']


# In[77]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[78]:


lr.fit(X_train,y_train)


# In[79]:


lr.intercept_


# In[80]:


pred = lr.predict(X_test)


# In[81]:


pred


# In[82]:


sns.distplot((y_test-pred),bins=50)


# In[83]:


from sklearn import metrics


# In[84]:


print("MAE:", metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[85]:


y_test_2d = np.array(y_test).reshape((-1,1))


# In[86]:


pred


# In[87]:


np.array(y_test)


# In[88]:


lr.score(X_test,y_test)


# In[89]:


from sklearn import linear_model
lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
lasso_reg.fit(X_train,y_train)


# In[90]:


lasso_reg.score(X_test,y_test)


# In[ ]:




