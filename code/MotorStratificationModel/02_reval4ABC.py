######################################################################
#
#import libraries and define functions
#
#
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
import random
import sys

#num_iter = int(sys.argv[1])

# define a plot for the stability that include saving and not displaying
def plot_metrics(cv_score, figsize=(20, 10)):
#  #   """
#  #   Function that plot the average performance (i.e., normalized stability) over cross-validation
#  #   for training and validation sets.
# #
#     Parameters
#     ----------
#   cv_score: dictionary
# figsize: tuple (width, height)
#     """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(cv_score['train'].keys()),
    [me[0] for me in cv_score['train'].values()],
    label='training set')
    ax.errorbar(list(cv_score['val'].keys()),
    [me[0] for me in cv_score['val'].values()],
    [me[1][1] for me in cv_score['val'].values()],
    label='validation set')
    plt.xticks([lab for lab in cv_score['train'].keys()])
    plt.xlabel('Number of clusters')
    plt.ylabel('Normalized stability')
    #plt.show()
    fig2save = "stabilityplot"+str(seed)
    plt.savefig(os.path.join(saving_path,fig2save))
    plt.close()



#############################################################################################################################

#seed_list = [num_iter]

seed_list = [90]
#############################################################################################################################
### load data and definepaths

## Load data.csv
#file_path = "/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/reval4server/data"
#saving_path ="/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/reval4server/results"

file_path = "/Users/vmandelli/Google Drive/PHD/projects/motorNDAR/results/REVAL"
saving_path = "/Users/vmandelli/Google Drive/PHD/projects/motorNDAR/results"

#file_path = "/media/DATA/vmandelli/ABC_proj/data"
#saving_path = "/media/DATA/vmandelli/ABC_proj/results"
data_path_name = "data4reval_200921.csv"
data = pd.read_csv(os.path.join(file_path,data_path_name),
                   sep=',',
                   header=0,
                   low_memory=False,
                  index_col="subject_id")
data.index.name = 'subjectkey'
data.head()



###########################################################################################################################
#
### start the real script

# define columns to use
col2clust = ["mabc2_manualdexterity_std","mabc2_aimingcatching_std","mabc2_balance_std"]
col2vect = ['study_id','module','sex']

# select only data3use
data2clust = data[col2clust]

# prepare the stratifying vector
new_vect = data[col2vect[0]] + data[col2vect[1]].astype("str")

#initialize empty saving matrices
metric_mat_2save = pd.DataFrame()
ncl_mat_2save = pd.DataFrame()
out_mat_2save = pd.DataFrame()

# random splitting
for s in seed_list:
    seed = s
    X_train, X_test, y_train, y_test = train_test_split(data2clust, new_vect, test_size=0.30, random_state=seed,stratify=new_vect)
    new_vect_strat_tr = y_train.to_frame()
    new_vect_strat_ts = y_test.to_frame()

    # Preprocessing : Scale and apply UMAP
    Scaler = StandardScaler()
    preproc = umap.UMAP(n_neighbors=20, min_dist=0, n_components=2, random_state=42)
    X_tr = pd.DataFrame(preproc.fit_transform(Scaler.fit_transform(X_train)), index=X_train.index)
    X_ts = pd.DataFrame(preproc.transform(Scaler.transform(X_test)), index=X_test.index)
    strat_vect = new_vect_strat_tr


    # grid search
    metric_df = pd.DataFrame()
    metric_dict ={'fold': [],"n_neigh":[],'ncl':[],'stab':[],'err':[]}
    clust = KMeans(random_state=42)
    vect_neigh = [2,3,5,6,10]
    vect_fold = [2]
    for f in vect_fold:
        for n in vect_neigh:
            logging.info(f"fold:{f} -- nighbors:{n}")
            knn = KNeighborsClassifier(n_neighbors=n)
            relval = FindBestClustCV(s=knn, c=clust, nfold=f, nclust_range=list(range(2,11,1)), nrand=100)
            metric, ncl= relval.best_nclust(X_tr,iter_cv=100, strat_vect = strat_vect)
            metric_dict['fold'].append(f)
            metric_dict['n_neigh'].append(n)
            metric_dict['ncl'].append(ncl)
            metric_dict['stab'].append(metric['val'][ncl][0])
            metric_dict['err'].append(metric['val'][ncl][1][1])

    metric_df = pd.DataFrame(metric_dict)


    # save metrics from grid search
    name2save_metric_df = "metric_gridsearch_df_"+ str(seed)
    metric_df.to_csv(os.path.join(saving_path,name2save_metric_df))


    # find out the most suitable metric
    mean_stab_err = list(metric_df[["stab","err"]].apply(np.mean,1))
    idx = mean_stab_err.index(min(mean_stab_err))


    ## REVAL
    # initialize classes
    knn = KNeighborsClassifier(n_neighbors=metric_df.loc[idx,'n_neigh'])
    clust = KMeans(random_state=42)
    relval = FindBestClustCV(s=knn, c=clust, nfold=2, nclust_range=list(range(2,11,1)), nrand=100)


    # train
    metric, ncl= relval.best_nclust(X_tr,iter_cv=100, strat_vect = strat_vect)


    # test
    out = relval.evaluate(X_tr, X_ts, ncl)


    # save outputs
    # metrics mat
    metric_mat = pd.DataFrame()
    metric_mat['cl'] = metric['train']
    metric_mat.index=metric_mat['cl']

    c0='stab'
    clusters = sorted(metric['train'].keys())
    row_values = list()
    for i in clusters:
        #metric_mat['cluster_solution'] = i
        metric_mat.loc[i,'stab_tr'] = [metric['train'][i][0]]
        metric_mat.loc[i,'stab_tr_up'] = [metric['train'][i][1][0]]
        metric_mat.loc[i,'stab_tr_low'] = [metric['train'][i][1][1]]
        metric_mat.loc[i,'stab_val'] = [metric['val'][i][0]]
        metric_mat.loc[i,'stab_val_up'] = [metric['val'][i][1][0]]
        metric_mat.loc[i,'stab_val_low'] = [metric['val'][i][1][1]]
        metric_mat.loc[i,'seed'] = seed

    # ncl mat
    ncl_mat = pd.DataFrame()
    ncl_mat.loc[1,'ncl'] = ncl
    ncl_mat.loc[1,'seed'] = seed


    # out mat
    out_mat= pd.DataFrame()
    out_mat.loc[1,'test_acc'] = out.test_acc
    out_mat.loc[1,'train_acc'] = out.train_acc
    out_mat.loc[1,'seed'] = seed


    ## save all

    name2save_metric = "metric_"+ str(seed)+".csv"
    metric_mat.to_csv(os.path.join(saving_path,name2save_metric))

    name2save_ncl = "ncl_"+ str(seed)+".csv"
    ncl_mat.to_csv(os.path.join(saving_path,name2save_ncl))

    name2save_out = "out_"+ str(seed)+".csv"
    out_mat.to_csv(os.path.join(saving_path,name2save_out))

    # metric mat
    #metric_mat_2save = metric_mat_2save.append(metric_mat)
    # ncl
    #ncl_mat_2save = ncl_mat_2save.append(ncl_mat)
    # accuracy
    #out_mat_2save = out_mat_2save.append(out_mat)


    # save plots
    # stability plot
    plot_metrics(metric,figsize=(20, 10))

    # umap plots
    ## save umap_plot
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    plt.scatter(X_tr[[0]], X_tr[[1]], color=[colors[lab] for lab in out.train_cllab])
    fig2save = "umap_train_plot"+str(seed)
    plt.savefig(os.path.join(saving_path,fig2save))
    plt.close()
    plt.scatter(X_ts[[0]], X_ts[[1]], color=[colors[lab] for lab in out.test_cllab])
    fig2save = "umap_test_plot"+str(seed)
    plt.savefig(os.path.join(saving_path,fig2save))
    plt.close()

    
    
    # save variables
    train =pd.DataFrame({ 'cluster': out.train_cllab,"TR_TS":"TR"})
    test = pd.DataFrame({ 'cluster': out.test_cllab,"TR_TS":"TS"})

    train.index =X_tr.index
    test.index =X_ts.index

    dataset_trts_cl = train.append(test)
    name2save_results = "reval_results_"+ str(seed)+".csv"
    dataset_trts_cl.to_csv(os.path.join(saving_path,name2save_results))
    
    
# save final matrices
# metric_mat
#metric_mat_2save.to_csv(os.path.join(saving_path,"metric_mat.csv"))
# ncl
#ncl_mat_2save.to_csv(os.path.join(saving_path,"ncl_mat.csv"))
# accuracy
#out_mat_2save.to_csv(os.path.join(saving_path,"out_mat.csv"))
