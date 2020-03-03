import csv
import shutil
import re

import numpy as np
import pandas as pd

import os
import sys

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)

from utils import imputedata, save_pkl, load_pkl
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

N_ROI = 100

'''
Match the MR data with cognitive tasks

Tedious basic information matching
'''

# loading file
df_cogtask = pd.read_csv('./data/raw/CS_commonTasks_N196.csv', header=0)
df_cogtask = df_cogtask.sort_values(by='SIDNO')
RNO = df_cogtask['RNO'].values
# convert data to numerical values; if error, return nan
df_cogtask = df_cogtask.apply(pd.to_numeric, errors='coerce')
df_cogtask['RNO'] = RNO

print('{:55}:{:5}'.format('Number of participants originally', df_cogtask.shape[0]))
print('Female:{}'.format(np.sum(df_cogtask['GENDER'])))

# reverse TS switch cost score (bigger = better)
df_cogtask['TS_SWITCHCOST'] = df_cogtask['TS_SWITCHCOST'] * -1


# verbal fluency contrast
df_cogtask['VF'] = df_cogtask['VFT_L_CpM'] - df_cogtask['VFT_C_CpM']

#DS average
df_cogtask['DS_MeanSpan'] = (df_cogtask['DS_NDigit_Backwards'] + df_cogtask['DS_NDigit_Forward']) / 2

# get variables for efficiency calculation
with open('./references/efficencylist.csv', 'r') as f:
    reader = csv.DictReader(f)
    eff_lst = []
    for row in reader:
        eff_lst.append(row)

shutil.copy('./references/VariableDescriptions_original.csv', './references/VariableDescriptions.csv')

# calculate efficency score
for e in eff_lst:
    df_cogtask[e['EFF']] =  - df_cogtask[e['RT']] / df_cogtask[e['ACC']]

    # add description to the variable note
    with open('./references/VariableDescriptions.csv', 'a') as f:
        f.write(e['EFF'] + ','
        + 'The reverse efficiency score of task ' + '_'.join(e['EFF'].split('_')[:-1]) + '\n')

# semantics module contrast (bigger = better)
df_cogtask['RJT_strength_EFF'] = df_cogtask['RJT_pw_Strong_EFF'] - df_cogtask['RJT_pw_Weak_EFF']
df_cogtask['RJT_modality_EFF'] = df_cogtask['RJT_pp_EFF'] - df_cogtask['RJT_ww_EFF']
df_cogtask['RJT_specificity_EFF'] = df_cogtask['PMT_Specific_EFF'] - df_cogtask['PMT_General_EFF']

# save this version
df_cogtask.to_csv('./data/interim/CS_Tasks_withEFF_{}.csv'.format(timestr))
df_cogtask.to_pickle('./data/interim/CS_Tasks_withEFF_{}.pkl'.format(timestr))

# drop cases with more than 10 missings in cognitive tasks- listwise
null_cases_per_subj = np.sum(pd.isnull(df_cogtask.iloc[:, 43:-6]).values, axis=1)
excludeIdx = np.where(null_cases_per_subj > 10)
excludeSIDNO = df_cogtask.iloc[df_cogtask.index[excludeIdx], 1].values
df_cogtask_sub = df_cogtask.drop(df_cogtask.index[excludeIdx])

# create a subset with selected variables
with open('./references/select_var.txt') as f:
    var_lst = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
var_lst = [x.strip() for x in var_lst] 
var_lst = ['AGE', 'GENDER', 'EDUCATION'] + var_lst

df_selected = df_cogtask_sub[['IDNO', 'SIDNO', 'RNO'] + var_lst]

# impute missing data as mean and outliers as mean+-2sd
dat = df_cogtask_sub[var_lst].values
dat = imputedata(dat, strategy='median')
dat = np.hstack((df_cogtask_sub.values[:, :3], dat))

df_impute = pd.DataFrame(dat, index=df_selected.index, columns=df_selected.columns)
idx = df_impute.index
print('{:55}:{:5}'.format('Number of participants selected', df_selected.shape[0]))
print('Female:{}'.format(np.sum(df_selected['GENDER'])))

# save
df_selected.to_csv('./data/interim/CS_Tasks_selectedvar_{}.csv'.format(timestr))
df_impute.to_csv('./data/interim/CS_Tasks_selectedvar_impute_{}.csv'.format(timestr))

# save as pkl as well
df_selected.to_pickle('./data/interim/CS_Tasks_selectedvar_{}.pkl'.format(timestr))
df_impute.to_pickle('./data/interim/CS_Tasks_selectedvar_impute_{}.pkl'.format(timestr))

'''
Get FC data
'''
# load FC data ID
path_FC_id = './data/interim/craddock100_fc_mat/DV_Conn_Cohort_List.csv'
path_FC_mat = './data/interim/craddock100_fc_mat/resultsROI_Condition001.mat'

import scipy.io as sio
FCID = pd.read_csv(path_FC_id)
FC_mat = sio.loadmat(path_FC_mat)

# match ID by R number
count = 0
fc_idx = []
for i in df_impute['RNO']:
    try:
        fc_idx.append(FCID.iloc[:, 0].tolist().index(i))
        count += 1

    except ValueError:
        pass

print('{} participants with imaging data that passes the quality check'.format(count))

RNO_selected = FCID.iloc[fc_idx, 0].values.tolist()

corr_mat = FC_mat['Z'][:, :N_ROI, fc_idx]
triu_inds = np.triu_indices(corr_mat.shape[0], 1)
corr_mat_vect = corr_mat[triu_inds]

region_labels = []
for n in FC_mat['names'][0]:
    num = re.findall('\d+', n[0])[-1]
    num = re.sub('^0+', '', num)
    region_labels.append(num)

reg_reg_names = [region_labels[a] + ' vs ' + region_labels[b] for (a,b) in zip(triu_inds[0], triu_inds[1])]

FC = {
    'FC_mat' : corr_mat,
    'FC_data' : corr_mat_vect.T,
    'ROI' : region_labels,
    'FC_labels' : reg_reg_names,
    'ID'   : RNO_selected

}

save_pkl(FC, './data/interim/Craddock100_FC_prepro_{}.pkl'.format(timestr))


'''
Get MWQ
'''

# load MWQ data
MWQ_raw = './data/raw/CS_MWQ_LabOnlineThoughtProbesScores_rescaled.csv'
df_MWQ = pd.read_csv(MWQ_raw, header=0).sort_values(by='SIDNO')
df_aggMean = pd.pivot_table(df_MWQ, values=list(df_MWQ.columns[7:]), index=['IDNO','SIDNO', 'RIDNO', 'session', 'nBack'], aggfunc=np.mean)
df_aggMean.reset_index(inplace=True)

# impute missing data as mean and outliers as median
dat = df_aggMean[df_aggMean.columns[4:]].values
dat = imputedata(dat, strategy='median')
dat = np.hstack((df_aggMean.values[:, :4], dat))
df_MWQ_impute = pd.DataFrame(dat, index=df_aggMean.index, columns=df_aggMean.columns)


# save
df_MWQ_impute.to_pickle('./data/interim/CS_MWQ_prepro_{}.pkl'.format(timestr))
df_MWQ_impute.to_csv('./data/interim/CS_MWQ_prepro_{}.csv'.format(timestr))

'''
get the participant with both the imaging and the task data
'''

CONFOUND_PATH = './data/raw/CS_MeanFD.csv'
MOT = pd.read_csv(CONFOUND_PATH, header=0).set_index('RNO').iloc[:206, 2:]

# put RS into a data frame
df_RS = pd.DataFrame(FC['FC_data'], columns=FC['FC_labels'])
df_RS['RNO'] = FC['ID']
df_RS = df_RS.set_index('RNO')

# match data by R number
lst_df = [df_impute.set_index('RNO'), MOT, df_RS]
df_master = pd.concat(lst_df, axis=1, join='inner')
region_labels = FC['ROI']

# save the master file
df_master.to_pickle('./data/interim/df_master_{}.pkl'.format(timestr))

print(
"Data path:\n\
    FC_path : ./data/interim/Craddock100_FC_prepro_{}.pkl\n \
    MOT_path :./data/raw/CS_MeanFD_{}.csv \n \
    data_path :./data/interim/CS_Tasks_selectedvar_impute_{}.pkl \n \
    prepro :./data/interim/df_master_{}.pkl' \n ".format(timestr)
)