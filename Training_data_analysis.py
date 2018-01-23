
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

# Read table

train_df = pd.read_table("./train.tsv", index_col = 'train_id', dtype = {'item_condition_id':CategoricalDtype(categories = [str(i) for i in range(1,6)], ordered = True), 'category_name':'category', 'brand_name': 'category', 'shipping':'category'})


# In[3]:

# Create columns for the first three hierarchical levels represented in the category name column.
# Levels greater than one contain the names of the parent levels with the current level, separeated by slashes

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

# The categories of the first category level

for i in train_df.category_oneLevel.cat.categories.values:
    print(i)


# In[5]:

# The path of each category in the second level

for i in train_df.category_twoLevel.cat.categories.values:
    print(i)


# In[6]:

# The first few rows of the training data
train_df.head()


# In[7]:

# Create column for (adult) clothing type that does not distinguish gender
# NaN's stand for all the items that are not clothing, or have no category (but could be clothing anyway)
# Note: This probably could have been dong without try-except

def get_clothing_type(elem):
    try:
        cat = elem['category_name'].split("/")
        if cat[0] in ['Men', 'Women']: return cat[1]
        else: return np.nan  
    except AttributeError: return np.nan # This error means that the category_name was NaN in the first place

train_df['clothing_type'] = train_df.apply(get_clothing_type, axis = 1).astype('category')


# In[8]:

# The first few rows of the training data with the new transformed columns

train_df.head(20)


# In[9]:

# Basic statistics about price, the only numerical category in the data

train_df.describe()


# In[10]:

# A logarithmically scaled histogram of the price distribution

plt.hist(train_df.price[train_df.price > 0], bins=10**np.linspace(0, 3, 25))
plt.title('Log scaled price distribution histogram')
plt.xscale('log') 
plt.show() 


# In[11]:

# This probability plot tests whether log-transformed prices are normally distributed.
# They seem to be heavy-tailed on the right, which is a pattern that mirrors pricing in financial markets (leptokurtosis)
# The heavy right tail is also visible in the histogram above
stats.probplot(np.log(train_df.price[train_df.price > 0]), dist = 'norm', plot = plt)


# In[12]:

# The number of items with each item condition category
train_df.item_condition_id.value_counts()


# In[13]:

# A bar plot of the number of items with each item condition category, graphical representation of the above cell
# I think pandas has a simplier method for a categorical series object, I'm not sure I needed to set this up myself in matplotlib

y_pos = np.arange(len(train_df.item_condition_id.value_counts()))
plt.bar(y_pos, train_df.item_condition_id.value_counts(), align='center', alpha=0.5)
plt.xticks(y_pos, train_df.item_condition_id.cat.categories.values)
plt.ylabel('Number of items')
plt.xlabel('Item condition number')


# In[33]:

# The next goal is to figure out, within each category, and then as a whole, the proportion of the error that can be eliminated by specifying the category.
# Based on intuition from visual inspection of the names of the categories, it seems that specifying the second level category plausibly eliminates much of the error.

# Need to consolidate logs so this function goes faster

'''
def get_group_stats(group_like): #Takes group or Series - but it needs to be made into two functions for each input type, because as it is you have to many if's
    
        log_group_like = group_like.transform(lambda y: np.log(y + 1))
    if isinstance(group_like, pd.core.groupby.SeriesGroupBy):
       # log_group_stats = log_group_st
    
    # Log ones need to be counted groupwise
    mean_raw = group_like.mean()
    mean_log = log_group_like.mean()
    std_log = log_group_like.std()
    sem_log = std_log/log_group_like.count()
    
    group_stats_dict = {'mean' : mean_raw, 'mean log': mean_log, 'std log' : std_log, 'sem log' : sem_log}
    
    if isinstance(group_like, pd.core.groupby.SeriesGroupBy):
        group_stats = pd.DataFrame(group_stats_dict)
    elif isinstance(group_like, pd.core.series.Series):
        group_stats = pd.Series(group_stats_dict)
    else:
        raise Exception('Must be Series or Grouped Series')
    
    return group_stats
''' 

new_df = train_df.drop('category_twoLevel', axis = 1).assign(category_twoLevel = train_df.category_twoLevel.cat.add_categories('NaN').fillna('NaN'))
new_df = new_df.assign(log_price = np.log(new_df.price + 1))

# Log price stats for whole dataset
price_stats = pd.Series({'n' : new_df.price.count(), 'mean' : new_df.price.mean(), 'mean log': new_df.log_price.mean(), 'std log' : new_df.log_price.std(), 'sem log' : new_df.log_price.std()/np.sqrt(new_df.log_price.count())})
#price_stats = get_group_stats(train_df.price)

# Log price stats for each category - must add transform
grouped_df = new_df.groupby('category_twoLevel')
stats_dict = {'n' : grouped_df.price.count(), 'mean' : grouped_df.price.mean(), 'mean log': grouped_df.log_price.mean(), 'std log' : grouped_df.log_price.std(), 'sem log' : grouped_df.log_price.std()/np.sqrt(grouped_df.log_price.count())}
category2_price_stats = pd.DataFrame(stats_dict)
#category2_price_stats = get_group_stats(train_df.drop('category_twoLevel', axis = 1).assign(category_twoLevel = train_df.category_twoLevel.cat.add_categories('NaN').fillna('NaN')).groupby('category_twoLevel').price)


# In[34]:

price_stats


# In[35]:

category2_price_stats


# In[36]:

# 


# In[39]:

# Planning to regress log price on the category names
# This needs to be changed to log price as is done with the rest of this notebook
cat_one_results = sm.ols('log_price ~ category_oneLevel', data = new_df).fit()
cat_one_results.summary()


# In[49]:

cat_one_results.params


# In[50]:

cat_one_results.bse


# In[58]:

# Assuming Gaussian distrubution (not completely sound, as shown by prior probability plot),
# Find the KL divergences for each category
cat_one_means = pd.Series(cat_one_results.params[0] + cat_one_results.params[1:])
#{'mean' : cat_one_results.params)
cat_one_sems = pd.Series(cat_one_results.bse[1:])
cat_one_stds = cat_one_sems


# In[56]:

# Divergences of distributions of means


#You want stds not sems


# In[ ]:



