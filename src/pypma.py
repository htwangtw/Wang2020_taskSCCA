#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Sparse Canonical Correlation Analysis from R package PMA.
H.-T. Wang

Calling the R library and wrap with scikit-learn like interface.

Reference:
http://cran.r-project.org/web/packages/PMA/PMA.pdf

Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition, with applications to sparse principal components and
canonical correlation analysis. Biostatistics, 10(3), 515–534. http://doi.org/10.1093/biostatistics/kxp008

Witten, D. M., & Tibshirani, R. J. (2009). Extensions of Sparse Canonical Correlation Analysis with Applications to Genomic Data.
Statistical Applications in Genetics and Molecular Biology, 8(1), 29. http://doi.org/10.2202/1544-6115.1470

'''
# define global variables
MODULE_PATH = '/groups/labs/semwandering/SCCA/src'

# import modules
import os
os.sys.path.append(MODULE_PATH)

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
from numpy.random import randint
from sklearn.model_selection import KFold, ParameterGrid

class SCCA(object):
    def __init__(self, C=[1, 1], penalty=['l1', 'l1'], \
            n_component=None, n_iter=100, verbose=False):
        self.C = C
        self.penalty = penalty
        self.n_component = n_component
        self.n_iter = n_iter
        self.verbose = verbose

    def fit(self, X, Y):
        numpy2ri.activate()
        rPMA = importr('PMA')
        typex, typez = _check_penalty_type(self.penalty)
        X, x_mean, x_std = _center_data(X)
        Y, y_mean, y_std = _center_data(Y)
        if self.n_component is None:
            self.n_component = np.min([X.shape[1], Y.shape[1]])
        out = rPMA.CCA(x=X, z=Y, K=self.n_component, \
                niter=self.n_iter, standardize=False, \
                typex=typex, typez=typez, \
                penaltyx=self.C[0], penaltyz=self.C[1], \
                trace=False)

        self.u = numpy2ri.ri2py(out[0])
        self.v = numpy2ri.ri2py(out[1])
        self._x_score, self._y_score = self.transform(X, Y)
        self._cancorr = _cancorr(X, Y, self.u, self.v)
        numpy2ri.deactivate()
        return self

    def predict(self):
        return None

    def transform(self, X, Y):
        X, x_mean, x_std = _center_data(X)
        Y, y_mean, y_std = _center_data(Y)
        return X.dot(self.u), Y.dot(self.v)

    def score(self, X, Y):
        return _cancorr(X, Y, self.u, self.v)

def permute(X, Y, permute=5, cutoff=1e-15, alpha=0.05):
    permute = permute
    cutoff = cutoff
    alpha = alpha
    param_grid = {'C_x': np.array(range(1, 11)) * 0.1,
                           'C_y': np.array(range(1, 11)) * 0.1}

    param_grid = ParameterGrid(param_grid)
    sig_mod = []
    for parameters in iter(param_grid):
        model = SCCA(C=[parameters['C_x'], parameters['C_y']],
                            penalty=['l1', 'l1'],
                            n_component=None, verbose=False)
        # fit the model on the full set
        model.fit(X, Y)
        d, u, v = model._cancorr, model.u, model.v

        # save for later
        d_sign = map(np.sign, d)
        u_sign = map(np.sign, u)
        v_sign = map(np.sign, v)

        # start bootstrapping
        bootsInd = randint(X.shape[0],size=(permute, X.shape[0]))

        # place holder for ordered results
        D_i = np.zeros((permute, model.n_component))
        U_i = np.zeros((permute, X.shape[1], model.n_component))
        V_i = np.zeros((permute, Y.shape[1], model.n_component))
        for i, I in enumerate(bootsInd):
            # get a bootstrap sample
            X_res = X[I,:]
            Y_res = Y[I,:]
            # fit the resample
            model.fit(X_res, Y_res)
            # reorder the resample canonical variate to match the original sample
            d_i, u_i, v_i = model._cancorr, model.u, model.v
            cor_ori_res = np.abs(np.corrcoef(v, v_i))[v.shape[0]:, :v.shape[0]]
            cor_max = np.max(cor_ori_res,0)
            idx_max = np.argmax(cor_ori_res,0)
            # make sure the reordered results shar the same signs as the reference
            res_di = d_i[idx_max]
            res_ui = u_i[:, idx_max]
            res_vi = v_i[:, idx_max]

            di_sign = map(np.sign, res_di)
            ui_sign = map(np.sign, res_ui)
            vi_sign = map(np.sign, res_vi)

            res_di = res_di * d_sign * di_sign
            res_ui = res_ui * u_sign * ui_sign
            res_vi = res_vi * v_sign * vi_sign
            print(res_ui.shape)
            # save the reordered results
            D_i[i, :] = res_di
            U_i[i, ...] = res_ui
            V_i[i, ...] = res_vi

        # calculate the p value on the canonical correlation of all variates
        p_j = np.sum(d > D_i, 0) / float(permute)
        sig = p_j < alpha

        # select the significant ones for this parameter set
        d, u, v, p = d[sig], u[:, sig], v[:, sig], p_j[sig]
        sig_mod.append[{'can_corr': d,
                        'u'       : u,
                        'v'       : v,
                        'p val'   : p,
                        'C_x'     : parameters['C_x'],
                        'C_y'     : parameters['C_y'],
                            }]
    # return the significant result of each parameter set, pass that to a cross validation test
    return sig_mod

class cvSCCA(SCCA):
    def __init__(self, folds=5, cutoff=1e-15):
        self.folds = folds
        self.cutoff = cutoff
        self.param_grid = {'C_x': np.array(range(1, 11)) * 0.1,
                           'C_y': np.array(range(1, 11)) * 0.1,
                           'K':   None}
    def train(self, X, Y):
        '''
            find the best penalty and number of components
            the best model is determined by the average canonical correlation
        '''
        self.param_grid['K'] = np.array(range(1, _max_n(X, Y) + 1))

        param_grid = ParameterGrid(self.param_grid)
        KF = KFold(n_splits=self.folds, shuffle=True, random_state=1)

        self.corr_lst = []
        best = 0
        best_model  = None
        for parameters in iter(param_grid):
            # learning parameters
            model = self.SCCA(C=[parameters['C_x'], parameters['C_y']], penalty=['l1', 'l1'], \
                                n_component=param_grid['K'], verbose=False)
            corr_lst = []
            lat_best = 0
            lat_best_model  = None
            for train_index, test_index in KF.split(X, Y):
                # gather train-test data
                X_train, Y_train = X[train_index, :], Y[train_index, :]
                X_test, Y_test = X[test_index, :], Y[test_index, :]
                model.fit(X_train, Y_train)

                d_j = np.mean(model.score(zscore(X_test), zscore(Y_test)))
                corr_lst.append(d_j)

                if d_j > lat_best:
                # select the best model of the current parameter set
                    lat_best_model = copy.deepcopy(model)
                    lat_best = d_j
            self.corr_lst.append(corr_lst)
            if lat_best > best:
                best_model = copy.deepcopy(lat_best_model)
                best = lat_best
        self.best_model = best_model.fit(X, Y)
        return self.best_model

#def _p_val(d_j, d_j_i):
#    p_val = np.mean(axis=)

def _check_penalty_type(penalty):
    rPenaltyType = []
    for p in penalty:
        if p == 'l1':
            rPenaltyType.append('standard')
        elif p == 'l2':
            rPenaltyType.append('ordered')
        else:
            rPenaltyType.append('standard')
            print('Penalty type not supported. Use default l1')
    return rPenaltyType

def _max_n(X, Y):
    return np.min([X.shape[1], Y.shape[1]])

def _center_data(X):
    mean = np.mean(X, axis=0)
    X -= mean
    std = np.std(X, axis=0, ddof=1)
    X /= std
    return X, mean, std

def _cancorr(X, Y, u, v):
    n = u.shape[1]
    X,_, _ = _center_data(X)
    Y,_,_ = _center_data(Y)
    return np.corrcoef(X.dot(u).T, Y.dot(v).T)[n:, 0:n].diagonal()

