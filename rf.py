import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from itertools import permutations, combinations
import dataprep as dp
from scipy.stats import norm
import math



########
# ICP
########


def getTestProbs(cube, labels, event_df, cluster_var, cluster, numTrees=100, envVar=None):
    
    
    
    #print("cluster: ", cluster)
    indx_val, =  np.where(event_df[cluster_var]==cluster)
    indx_train = np.array(list(set(np.arange(0, cube.shape[0])).difference(indx_val)))
    
    X_train = cube[indx_train,:]
    X_val = cube[indx_val,:]
    y_train = labels[indx_train]
    y_val = labels[indx_val]
    
    if envVar is not None:
        envVar_train = envVar[indx_train,]
        envVar_val = envVar[indx_val,]
        X_train = np.concatenate([X_train, envVar_train], axis=1)
        X_val = np.concatenate([X_val, envVar_val], axis=1)
    
    event_df_train = event_df.iloc[indx_train]
    event_df_val = event_df.iloc[indx_val]
    
    clf = RandomForestClassifier(n_estimators=numTrees, max_depth=10, class_weight="balanced_subsample", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_val)
    return y_pred[:,1]

def getTestProbsWrapper(cube, labels, event_df, cluster_var, indxIncl, posts,  numTrees=100, envVar=None):
    
    if posts is not None: 
        indxIncl = np.array(dp.flatten([np.arange(posts[i], posts[i+1]).tolist() for i in indxIncl]))
    
    clusts = np.sort(np.unique(event_df[cluster_var]))
    y_preds = [getTestProbs(cube[:,indxIncl], labels, event_df, cluster_var, i, numTrees=numTrees, envVar=envVar) for i in clusts]
    y_pred = np.ones(cube.shape[0])*-1
    for i in clusts:
        indx_val, =  np.where(event_df[cluster_var]==i)
        y_pred[indx_val] = y_preds[i]
    return y_pred


def equalAUC_hypTest(y_pred_noE, y_pred_E, labels):
    # without E
    y_pred1 = y_pred_noE[labels==1]
    y_pred0 = y_pred_noE[labels==0]
    Phi = (y_pred1[:,None] > y_pred0[None,:])
    VT_noE_1 = (1/(y_pred0.shape[0]-1))*np.apply_along_axis(np.sum, 1, Phi)
    VT_noE_0 = (1/(y_pred1.shape[0]-1))*np.apply_along_axis(np.sum, 0, Phi)
    A_noE = (np.sum(VT_noE_1)/VT_noE_1.shape[0]+np.sum(VT_noE_0)/VT_noE_0.shape[0])/2
    #print("auc without E:",A_noE)
    ST_noE_1 = (VT_noE_1-A_noE)**2
    ST_noE_1 = np.sum(ST_noE_1)/(ST_noE_1.shape[0]-1)
    ST_noE_0 = (VT_noE_0-A_noE)**2
    ST_noE_0 = np.sum(ST_noE_0)/(ST_noE_0.shape[0]-1)
    VA_noE = (ST_noE_1/VT_noE_1.shape[0])+(ST_noE_0/VT_noE_0.shape[0])
    # with E
    y_pred1 = y_pred_E[labels==1]
    y_pred0 = y_pred_E[labels==0]
    Phi = (y_pred1[:,None] > y_pred0[None,:])
    VT_E_1 = (1/(y_pred0.shape[0]-1))*np.apply_along_axis(np.sum, 1, Phi)
    VT_E_0 = (1/(y_pred1.shape[0]-1))*np.apply_along_axis(np.sum, 0, Phi)
    A_E = (np.sum(VT_E_1)/VT_E_1.shape[0]+np.sum(VT_E_0)/VT_E_0.shape[0])/2
    #print("auc with E: ",A_E)
    ST_E_1 = (VT_E_1-A_E)**2
    ST_E_1 = np.sum(ST_E_1)/(ST_E_1.shape[0]-1)
    ST_E_0 = (VT_E_0-A_E)**2
    ST_E_0 = np.sum(ST_E_0)/(ST_E_0.shape[0]-1)
    VA_E = (ST_E_1/VT_E_1.shape[0])+(ST_E_0/VT_E_0.shape[0])
    # Covariance
    ST_EnoE_1  = (VT_noE_1-A_noE)*(VT_E_1-A_E)
    ST_EnoE_1 = np.sum(ST_EnoE_1)/(ST_EnoE_1.shape[0]-1)
    ST_EnoE_0  = (VT_noE_0-A_noE)*(VT_E_0-A_E)
    ST_EnoE_0 = np.sum(ST_EnoE_0)/(ST_EnoE_0.shape[0]-1)
    COV_EnoE = (ST_EnoE_1/VT_E_1.shape[0])+(ST_EnoE_0/VT_E_0.shape[0])
    # Statistic
    V_noE_E = VA_noE + VA_E - 2*COV_EnoE
    z = (A_E-A_noE)/np.sqrt(V_noE_E)
    # H0: with E is not better -> A_E-A_noE is not large
    pval1tail = 1-norm.cdf(z)
    # H0: neither model is better
    pval2tail = (1-norm.cdf(np.abs(z)))+norm.cdf(-np.abs(z))
    res = {"stat":z, "pval_1tail":pval1tail, "pval_2tail":pval2tail,"auc_E":A_E, "auc_noE":A_noE}
    return res


def getHypWrapper(cube, labels, envVar, event_df, cluster_var, indxIncl, posts, numTrees=100):
    y_pred_noE = getTestProbsWrapper(cube, labels, event_df, cluster_var, indxIncl, posts, numTrees=numTrees)
    y_pred_E = getTestProbsWrapper(cube, labels, event_df, cluster_var, indxIncl, posts, numTrees=numTrees, envVar=envVar)
    res = equalAUC_hypTest(y_pred_noE, y_pred_E, labels)
    return res

