# function 1
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from reval.best_nclust_cv import FindBestClustCV
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier # example 1
from sklearn.linear_model import LogisticRegression # example 2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap

def scaling_umap(TR,TS,MOD,STUD):
    """
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import umap
    import pandas as pd
    # merge stratify vectors into one (only for train)
    strat_vect = STUD['study_id'] + MOD['module'].astype("str")
    
    # scale and apply umap
    Scaler = StandardScaler()
    preproc = umap.UMAP(n_neighbors=20, min_dist=0, n_components=2, random_state=42)
    
    X_tr_prep = pd.DataFrame(preproc.fit_transform(Scaler.fit_transform(TR)), index=TR.index)
    X_ts_prep = pd.DataFrame(preproc.transform(Scaler.transform(TS)), index=TS.index)
    
    return X_tr_prep,X_ts_prep,strat_vect




## function 2
def Gridsearch(X_tr,strat_vect):

    
    # GRID SEARCH
    
    # initialize impty saving matrices
    #metric_df = pd.DataFrame()
    metric_dict ={'fold': [],"n_neigh":[],'ncl':[],'stab':[],'err':[]}
    
    
    # define params to test
    # KMEANS
    clust = KMeans(random_state=42)
    vect_hyper = [2,3,5,6,10]
    # Herarchical Clutering
    #clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')
    #vect_hyper = [10,15,20,25]
    
    vect_fold = [2]
    
    # run the grid search
    for f in vect_fold:
        for n in vect_hyper:
            logging.info(f"fold:{f} -- nighbors:{n}")
            clf = KNeighborsClassifier(n_neighbors=n)
            #clf = RandomForestClassifier(n_estimators=n)
            relval = FindBestClustCV(s=clf, c=clust, nfold=f, nclust_range=list(range(2,5,1)), nrand=100)
            metric, ncl = relval.best_nclust(X_tr,iter_cv=100, strat_vect = strat_vect) 
            metric_dict['fold'].append(f)
            metric_dict['n_neigh'].append(n)
            metric_dict['ncl'].append(ncl)
            metric_dict['stab'].append(metric['val'][ncl][0])
            metric_dict['err'].append(metric['val'][ncl][1][1])

    metric_df = pd.DataFrame(metric_dict)

    return metric_df


# function 3

def RunREVAL(X_tr,X_ts,strat_vect,n):
    # Initialize classes
    # classifier
    clf = KNeighborsClassifier(n_neighbors= n)      #KNN
    # n=15
    # clf = RandomForestClassifier(n_estimators=15) #RF
    
    # cluster algorhythm
    clust = KMeans(random_state=42)                                         #Kmeans
    #clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')     #hierarchical
    
    # initialize the class
    relval = FindBestClustCV(s=clf, c=clust, nfold=2, nclust_range=list(range(2,11,1)), nrand=100)
    
    # train
    metric, ncl= relval.best_nclust(X_tr,iter_cv=100, strat_vect = strat_vect)
    print(metric)
    print(ncl)
    
    #test
    out = relval.evaluate(X_tr, X_ts, ncl) # riscrivi NCL qui! il best!
    logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")

    return relval,metric,ncl,out
    
    