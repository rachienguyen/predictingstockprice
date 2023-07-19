#!/usr/bin/env python
# coding: utf-8

# In[29]:


import yfinance as yf


# In[30]:


apple=yf.Ticker("aapl")


# In[31]:


apple=apple.history(period="max")


# In[32]:


apple


# In[33]:


apple.plot.line(y="Close", use_index=True)


# In[34]:


apple.plot.line(y="Close", use_index=True)


# In[35]:


del apple["Dividends"]
del apple["Stock Splits"]


# In[36]:


apple


# In[37]:


apple["Tomorrow"]=apple["Close"].shift(-1)


# In[38]:


apple


# In[39]:


apple["Target"]=(apple["Tomorrow"]>apple["Close"]).astype(int)


# In[40]:


apple


# In[41]:


apple=apple.loc["2000-01-01":].copy()
apple


# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[43]:


model= RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)


# In[44]:


train=apple.iloc[:-100]


# In[45]:


test=apple.iloc[-100:]


# In[46]:


predictors=["Close","Volume","Open","High","Low"]


# In[47]:


model.fit(train[predictors],train["Target"])
RandomForestClassifier(min_samples_split=100,random_state=1)


# In[48]:


from sklearn.metrics import precision_score
preds=model.predict(test[predictors])
preds


# In[49]:


import pandas as pd
preds=pd.Series(preds,index=test.index)
precision_score(test["Target"],preds)


# In[50]:


combine=pd.concat([test["Target"],preds],axis=1)
combine.plot()


# In[51]:





# In[52]:





# In[55]:





# In[ ]:




