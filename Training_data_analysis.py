
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import scipy.stats as stats
import statsmodels.formula.api as sm

print(pd.__version__)

get_ipython().magic('matplotlib inline')


# In[2]:

train_df = pd.read_table("./train.tsv", index_col = 'train_id', dtype = {'item_condition_id':CategoricalDtype(categories = [str(i) for i in range(1,6)], ordered = True), 'category_name':'category', 'brand_name': 'category', 'shipping':'category'})


# In[3]:

a, b, c, d = train_df['category_name'].str.split("/", 3).str
category_oneLevel = a
category_twoLevel = a + "/" + b
category_threeLevel = a + "/" + b + "/" + c
train_df['category_oneLevel'] = category_oneLevel
train_df['category_twoLevel'] = category_twoLevel
train_df['category_threeLevel'] = category_threeLevel
for col in ['category_oneLevel', 'category_twoLevel', 'category_threeLevel']:
    train_df[col] = train_df[col].astype('category')


# In[4]:

for i in train_df.category_oneLevel.cat.categories.values:
    print(i)


# In[5]:

for i in train_df.category_twoLevel.cat.categories.values:
    print(i)


# In[6]:

train_df.head()


# In[7]:

# Create column for (adult) clothing type that does not distinguish gender - You will just have NaN's for all the items that are not clothing

def get_clothing_type(elem):
    try:
        cat = elem['category_name'].split("/")
        if cat[0] in ['Men', 'Women']: return cat[1]
        else: return np.nan  
    except AttributeError: return np.nan # This error means that the category_name was NaN in the first place

train_df['clothing_type'] = train_df.apply(get_clothing_type, axis = 1).astype('category')


# In[8]:

train_df.head(20)


# In[9]:

train_df.describe()


# In[10]:

plt.hist(train_df.price[train_df.price > 0], bins=10**np.linspace(0, 3, 25))
plt.title('Log transformed price distribution histogram')
plt.xscale('log') 
plt.show() 


# In[11]:

# This probability plot tests whether log-transformed prices are normally distributed.
# They seem to be heavy-tailed on the right, which is a pattern that mirrors pricing in financial markets (leptokurtosis)
# The heavy right tail is also visible in the histogram above
stats.probplot(np.log(train_df.price[train_df.price > 0]), dist = 'norm', plot = plt)


# In[12]:

train_df.item_condition_id.value_counts()


# In[13]:

train_df.item_condition_id.cat.categories.values


# In[14]:

y_pos = np.arange(len(train_df.item_condition_id.value_counts()))
plt.bar(y_pos, train_df.item_condition_id.value_counts(), align='center', alpha=0.5)
plt.xticks(y_pos, train_df.item_condition_id.cat.categories.values)
plt.ylabel('Number of items')
plt.xlabel('Item condition number')


# In[ ]:

# The next goal is to figure out, within each category, and then as a whole, the proportion of the RSE that can be explained the category.
# Based on intuition from visual inspection, it seems that the second level category plausibly separates items into separate price buckets.


# In[15]:

#train_df.category_twoLevel.cat.categories.values
# This needs to be changed to log price
result = sm.ols(formula = 'price ~ category_twoLevel', data = train_df).fit()


# In[ ]:



