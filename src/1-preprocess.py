#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd -q ~/TaskSCCA_craddock/


# In[2]:


import os
import re

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from nilearn.plotting import cm

import pandas as pd

from scipy.stats.mstats import zscore
from scipy.stats import percentileofscore

from src.data_cleaning import clean_confound, mad, select_mad_percentile
from src.visualize import set_text_size
from src.utils import unflatten

sns.set_style({"font.sans-serif": ["Arial"]})
sns.set_context('paper', font_scale=1.5)


# In[3]:


path_master = 'data/interim/df_master_p178.pkl'

df_master = pd.read_pickle(path_master)

X = df_master.iloc[:, 25:].apply(pd.to_numeric, errors='coerce')
Y = df_master.iloc[:, 5:18].apply(pd.to_numeric, errors='coerce')
confound = df_master.loc[:, ['AGE', 'GENDER', 'MeanFD_Jenkinson']].apply(pd.to_numeric, errors='coerce')

COG_labels = df_master.columns[5:18]

X = X.values
Y = Y.values
confound = confound.values


# # Dimension reduction on the functional connectivity

# In[4]:


fc_mad = mad(X)
mad_percent = np.array([percentileofscore(fc_mad, a, 'rank') for a in fc_mad])
sort_idx = np.argsort(-fc_mad)
fc_sort = fc_mad[sort_idx]


# In[5]:


fc_mean = X.mean(axis=0)
fc_95, _ = select_mad_percentile(fc_mean, fc_mad, 95)
fc_90, _ = select_mad_percentile(fc_mean, fc_mad, 90)
fc_75, _ = select_mad_percentile(fc_mean, fc_mad, 75)

full_mat = [unflatten(fc_mean), unflatten(fc_75), unflatten(fc_90), unflatten(fc_95)]
titles = ['Full matix', '75% matix', '90% matix', '95% matix']


# In[6]:


fig, ax = plt.subplots(figsize=(6,4))

#ax.plot(fc_sort)
ax.fill_between(range(0,4950), 0, fc_sort)
ax.fill_between(range(0,4950), 0, fc_sort, where= fc_sort > np.percentile(fc_mad, 50))
ax.fill_between(range(0,4950), 0, fc_sort, where= fc_sort > np.percentile(fc_mad, 75))
ax.fill_between(range(0,4950), 0, fc_sort, where= fc_sort > np.percentile(fc_mad, 90))
ax.fill_between(range(0,4950), 0, fc_sort, where= fc_sort > np.percentile(fc_mad, 95))
plt.xlim((1, 4950))
plt.ylim((0, 0.27))
plt.xticks([1, np.percentile(range(0,4950), 5), np.percentile(range(0,4950), 10),
             np.percentile(range(0,4950), 25),np.percentile(range(0,4950), 50), 4950],
           rotation=90
            )

ax.annotate('95%', xy=(np.percentile(range(0,4950), 2), 0.23))
ax.annotate('90%', xy=(np.percentile(range(0,4950), 7), 0.18))
ax.annotate('75%', xy=(np.percentile(range(0,4950), 15), 0.16))
ax.annotate('50%', xy=(np.percentile(range(0,4950), 35), 0.14))

plt.xlabel('Functional connectivity edges')
plt.ylabel('Median absolute deviation')
plt.savefig('reports/figures/mad_percentile.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close()


# In[7]:


def get_primary(x):
    list_name = re.findall("[a-zA-Z]+", x)
    if len(list_name) == 1:
        prim = list_name[0]

    elif list_name[0] == 'None':
            prim = list_name[1]
    else:
        prim = list_name[0]
    return prim

def sort_by_yeo7(mat):

    def get_ticks():
        ticks = [1]
        ticklabels = ['Default']
        for i in range(100):
            cur = label_names_yeo7.iloc[i, 3]
            if ticklabels[-1] != cur:
                ticks.append(i + 1)
                ticklabels.append(cur)
        return ticks

    label_names = pd.read_csv('references/scorr05_2level_names_100.csv')

    label_names['Yeo7'] = label_names['Yeo-Krienen 7 networks'].apply(get_primary)
    label_names = label_names.sort_values('Yeo7')
    label_names_yeo7 = label_names.iloc[:, [0, 1, -1]].reset_index()

    tmp = pd.DataFrame(mat, index=range(1, 101), columns=range(1, 101))
    idx = label_names_yeo7.index.tolist()
    reorder = tmp.reindex(index=idx, columns=idx)
    ticks = get_ticks()
    ticklabels = ['DMN', 'DAN', 'FPN', 'LIM', 'None', 'S-M', 'VAN', 'VIS']
    return reorder.values, ticks, ticklabels


# In[8]:


# plotting
fig, axarr= plt.subplots(2, 2, figsize=(10, 10))

for i, (mat, title) in enumerate(zip(full_mat, titles)):
    loc_x, loc_y = np.unravel_index(i, (2,2))
    ax_cur = axarr[loc_x, loc_y]
    ax_cur.set_title(title)
#     hm = ax_cur.imshow(mat, vmin=-0.4, vmax=0.4, cmap='cold_hot')
#     fig.colorbar(hm, ax=ax_cur)
    mat, ticks, ticklabels = sort_by_yeo7(mat)
    sns.heatmap(mat, center=0,
                square=True, annot=False,
                cmap='cold_hot', ax=ax_cur)

    ax_cur.hlines(ticks[1:], *ax_cur.get_xlim(), color='w', lw=0.5)
    ax_cur.vlines(ticks[1:], *ax_cur.get_ylim(), color='w', lw=0.5)

    ax_cur.set_xticks(ticks)
    ax_cur.set_xticklabels(ticklabels)
    ax_cur.set_yticks(ticks)
    ax_cur.set_yticklabels(ticklabels)

plt.savefig('reports/figures/FC_mat.png', dpi=300, transparent=True)

plt.show()
plt.close()


# # Preprocess the dataset

# In[9]:


# select 95% of the edges
# _, mask_95 = select_mad_percentile(fc_mean, fc_mad, 95)

# X_masked = X[:, mask_95]

# # clean and z score the data
# X_clean, Y_clean, confmat = clean_confound(X_masked, Y, confound)

# save the cleaned data and the mask for the late analysis and reconstructing niftis
# np.save('data/processed/X_clean', X_clean)
# np.save('data/processed/Y_clean', Y_clean)
# np.save('data/processed/X_mask', mask_95)
# np.save('data/processed/confound', confmat)

