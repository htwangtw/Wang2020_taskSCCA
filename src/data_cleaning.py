import numpy as np
import os
from scipy.stats.mstats import zscore
from nilearn.signal import clean


def clean_confound(RS, COG, confmat):
    '''
    clean things and zscore things
    '''

    # regress out confound
    z_confound = zscore(confmat)
    # squared measures to help account for potentially nonlinear effects of these confounds
    z2_confound = z_confound ** 2
    conf_mat = np.hstack((z_confound, z2_confound))

    # Handle nan in z scores
    conf_mat = np.nan_to_num(conf_mat)

    # clean signal
    RS_clean = clean(zscore(RS), confounds=conf_mat, detrend=False, standardize=False)
    COG_clean = clean(zscore(COG), confounds=conf_mat, detrend=False, standardize=False)

    return RS_clean, COG_clean, conf_mat


def mad(X):
    '''
    Median absolute deviation
    source: https://en.wikipedia.org/wiki/Median_absolute_deviation
    '''
    med = np.median(X, axis=0)
    return np.median(np.abs(X - med), axis=0)


def select_mad_percentile(X, X_mad, n):
    '''
    select features based on the MAD percentile
    Return a data mask for applying on the full dataset

    '''
    boo_mad = (X_mad > np.percentile(X_mad, n))
    return X * boo_mad.astype(int), boo_mad
