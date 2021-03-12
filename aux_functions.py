import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from statistics import mode

############################################################
###########       DATA MANIPULATION      ###################
############################################################

def standardize(data):
    means = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - means) / std
    return data

def feature_extraction(mat, method="all"):
    data = np.zeros((mat.shape[0], 18))
    pca = PCA(n_components=6)
    pca.fit(mat)
    data[:,:6] = pca.transform(mat)
    tsne = TSNE(n_components=6, method='exact')
    data[:,6:12] = tsne.fit_transform(mat)
    embedding = Isomap(n_components=6)
    data[:,12:18] = embedding.fit_transform(mat)
    if method == "PCA":
        return data[:,:6]
    elif method == "TSNE":
        return data[:,6:12]
    elif method == "Isomap":
        return data[:,12:]
    np.savetxt("feature_extraction.txt", data)
    print("Saved feature extraction to file: feature_extraction.txt")
    return data


def feature_selection(data, labels, n_feat):
    f,prob = f_classif(data, labels)
    k_best = SelectKBest(f_classif,k=n_feat)
    k_best.fit(data,labels)
    X_new = k_best.transform(data)
    features = k_best.get_support(indices=True)
    return f, X_new, features

############################################################
###########       METRICS FUNCTIONS       ##################
############################################################

def precision_score(tp, fp):            
    return tp/(fp+tp)


def recall_score(tp, fn):
    return tp/(fn+tp)


def f1_score(prec,rec):
    return (2*prec*rec)/(prec+rec)


def rand_index(tp, tn, n):
    return (tp+tn)/(n*(n-1)/2)


def count_metrics(true_labels, pred_labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(true_labels)):
        for j in range(i+1,len(true_labels)):
            if true_labels[i] == true_labels[j] and pred_labels[i] == pred_labels[j]:
                tp += 1
            if true_labels[i] != true_labels[j] and pred_labels[i] != pred_labels[j]:
                tn += 1
            if true_labels[i] != true_labels[j] and pred_labels[i] == pred_labels[j]:
                fp += 1
            if true_labels[i] == true_labels[j] and pred_labels[i] != pred_labels[j]:
                fn += 1
                
    return tp, fp, tn, fn

def count_metrics_biss(true_labels, pred_labels, spots):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(spots)):
        for j in range(i+1,len(spots)):
            if true_labels[spots[i]] == true_labels[spots[j]] and pred_labels[spots[i]] == pred_labels[spots[j]]:
                tp += 1
            if true_labels[spots[i]] != true_labels[spots[j]] and pred_labels[spots[i]] != pred_labels[spots[j]]:
                tn += 1
            if true_labels[spots[i]] != true_labels[spots[j]] and pred_labels[spots[i]] == pred_labels[spots[j]]:
                fp += 1
            if true_labels[spots[i]] == true_labels[spots[j]] and pred_labels[spots[i]] != pred_labels[spots[j]]:
                fn += 1
                
    return tp, fp, tn, fn

def evaluate_biss_kmeans(data, lbl, pred_lbl):
    spots = []
    for i in range(len(lbl)):
        if lbl[i]>0:
            spots.append(i)
    ##s = silhouette_score(data[pred_lbl!=-1,:], pred_lbl[pred_lbl!=-1])
    tp, fp, tn, fn = count_metrics_biss(lbl,pred_lbl,spots)
    p = precision_score(tp,fp)
    r = recall_score(tp,fn)
    f_1 = f1_score(p,r)
    ri = rand_index(tp,tn,len(spots))
    ##ar = adjusted_rand_score(lbl[lbl>0],pred_lbl[lbl>0])
    s=0
    ar=0
    return s, f_1, p, r, ri, ar

def evaluate(data, lbl, pred_lbl):
    unique_labels = set(pred_lbl)
    if len(unique_labels) <= 2:
        s = 0
        p = 0
        r = 0
        f_1 = 0
        ri = 0
        ar = 0
    else:
        s = silhouette_score(data[pred_lbl!=-1,:], pred_lbl[pred_lbl!=-1])
        tp, fp, tn, fn = count_metrics(lbl[lbl>0],pred_lbl[lbl>0])
        p = precision_score(tp,fp)
        r = recall_score(tp,fn)
        f_1 = f1_score(p,r)
        ri = rand_index(tp,tn,len(lbl[lbl>0]))
        ar = adjusted_rand_score(lbl[lbl>0],pred_lbl[lbl>0])
    return s, f_1, p, r, ri, ar

############################################################
###########       CLUSTERING FUNCTIONS       ###############
############################################################

def k_means_clustering(n, data):
    k_means = KMeans(n_clusters=n).fit(data)
    k_means_labels = k_means.predict(data)
    return k_means_labels
    

def dbscan_clustering(eps, data):
    db_labels = DBSCAN(eps=eps, min_samples=5).fit_predict(data)
    return db_labels

def agglomerative_clustering(n, data):
    agg_labels = AgglomerativeClustering(n_clusters=n).fit_predict(data)
    return agg_labels

def bissecting_kmeans_clustering(t_data, feats, n_clusters):
    t_aux = t_data
    t_data_comb = t_data
    result = [[] for i in range(t_data.shape[0])]
    record = [True for i in range(t_data.shape[0])]
    clusters = []
    while len(clusters) < n_clusters:
        k_means_labels = k_means_clustering(2,t_data_comb)
        idx = 0
        for place in range(len(record)):
            if record[place]:
                result[place].append(k_means_labels[idx])
                idx += 1
        
        stats = [tuple(x) for x in result]
        freq = mode(stats)
        for place in range(len(record)):
            if(result[place] == list(freq)):
                record[place] = True
            else:
                record[place] = False
        
        t_data_comb = t_aux[record,:]
        clusters = set(stats)
    
    return result

############################################################
###########       PLOT FUNCTIONS       #####################
############################################################
pix = 500

def plot_2d(X,y,file_name):
    plt.figure(figsize=(7,7))
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='black', alpha=0.5)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='red', alpha=0.5)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='green', alpha=0.5)
    plt.plot(X[y==3,0], X[y==3,1],'o', markersize=7, color='blue', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_name,dpi=200,bbox_inches='tight')
    plt.show()
    plt.close()


def plot_3d(data, labels, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[labels==0,0], data[labels==0,1], data[labels==0,2], c='white', s=10)
    ax.scatter(data[labels==1,0], data[labels==1,1], data[labels==1,2], c='red', s=10)
    ax.scatter(data[labels==2,0], data[labels==2,1], data[labels==2,2], c='green', s=10)
    ax.scatter(data[labels==3,0], data[labels==3,1], data[labels==3,2], c='blue', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(filename,dpi=200,bbox_inches='tight')
    plt.show()
    plt.close()


def plot_elbow(data, feats):
    neighbors = KNeighborsClassifier(n_neighbors=5)
    Y = np.zeros(data.shape[0])
    neighbors.fit(data, Y)
    
    dist = neighbors.kneighbors(return_distance=True)[0]
    arr = []
    for m in dist:
        arr.append(m[4])
    arr = np.array(arr)
    arr = np.sort(arr)
    arr = arr[::-1]
    plt.figure(figsize=(12,8))
    plt.plot(range(len(arr)),arr,'-b')
    plt.legend(["Error Train"])
    plt.title(feats)
    plt.show()
    plt.close()


def plot_comb(name, filename, arr, sil_score, f1_score, 
              precision, recall, rand_index, adjusted_rand):
    plt.figure(figsize=(7,7))
    plt.title(name)
    plt.plot(arr,sil_score,"-r")
    plt.plot(arr,f1_score,"-b")
    plt.plot(arr,precision,"-g")
    plt.plot(arr,recall,"-m")
    plt.plot(arr,rand_index,"-c")
    plt.plot(arr,adjusted_rand,"-k")
    plt.legend(["Silhouette","F1","Precision","Recall","Rand Index",
                "Adjusted Rand Score"])
    plt.savefig(filename, dpi=pix)
    plt.show()

def plot_kmeans_metrics(t_data_comb, labels, feats):
    sil = []
    f1 = []
    prec = []
    rec = []
    rand_idx = []
    ars = []
    
    for n in range(2,11):
        k_means_labels = k_means_clustering(n,t_data_comb)
        
        s, f_1, p, r, ri, ar = evaluate(t_data_comb, labels, k_means_labels)
        
        sil.append(s)
        f1.append(f_1)
        prec.append(p)
        rec.append(r)
        rand_idx.append(ri)
        ars.append(ar)
    
    title = "K-MEANS: " + str(feats)
    filename = "ClusteringAnalysis/KMeans_" + str(len(feats)) + "feats"
    plot_comb(title, filename, range(2,11),sil,f1,prec,rec,rand_idx,ars) 

def plot_bissecting_kmeans_metrics(t_data_comb, labels, feats):
    sil = []
    f1 = []
    prec = []
    rec = []
    rand_idx = []
    ars = []
    
    for n in range(2,11):
        res = bissecting_kmeans_clustering(t_data_comb, feats, n)
        pred = [tuple(x) for x in res]
        s, f_1, p, r, ri, ar = evaluate_biss_kmeans(t_data_comb, labels, pred)
        
        sil.append(s)
        f1.append(f_1)
        prec.append(p)
        rec.append(r)
        rand_idx.append(ri)
        ars.append(ar)
        res = []
    
    title = "Biss_K-MEANS: " + str(feats)
    filename = "ClusteringAnalysis/BissectingKMeans_" + str(len(feats)) + "feats"
    plot_comb(title, filename, range(2,11),sil,f1,prec,rec,rand_idx,ars)

def plot_dbscan_metrics(t_data_comb, labels, feats, epsilon, d_eps):
    sil = []
    f1 = []
    prec = []
    rec = []
    rand_idx = []
    ars = []
    
    best_rand_index = -1
    arr = np.arange(epsilon-d_eps, epsilon+d_eps, 0.01)
    
    for eps in arr:
    
        db_labels = dbscan_clustering(eps,t_data_comb)
        
        s, f_1, p, r, ri, ar = evaluate(t_data_comb,labels,db_labels)
        
        sil.append(s)
        f1.append(f_1)
        prec.append(p)
        rec.append(r)
        rand_idx.append(ri)
        ars.append(ar)
        if p > best_rand_index:
            best_rand_index = p
            best_eps = eps
        
    title = "DBSCAN: " + str(feats) + " best_eps: " + str(best_eps)
    filename = "ClusteringAnalysis/DBSCAN_" + str(len(feats)) + "feats" 
    plot_comb(title,filename,arr,sil,f1,prec,rec,rand_idx,ars)

def plot_agglomerative_metrics(t_data_comb, labels, feats):
    sil = []
    f1 = []
    prec = []
    rec = []
    rand_idx = []
    ars = []
    
    for n in range(2,11):
        agg_labels = agglomerative_clustering(n,t_data_comb)
        
        s, f_1, p, r, ri, ar = evaluate(t_data_comb, labels, agg_labels)
        
        sil.append(s)
        f1.append(f_1)
        prec.append(p)
        rec.append(r)
        rand_idx.append(ri)
        ars.append(ar)
    
    title = "Agglomerative: " + str(feats)
    filename = "ClusteringAnalysis/Agglomerative_" + str(len(feats)) + "feats"
    plot_comb(title, filename, range(2,11),sil,f1,prec,rec,rand_idx,ars)

def scatter_plot(t_data, feats, colors):
    df = pd.DataFrame(t_data[:,feats], columns=["PCA2","PCA3","TSNE1",
                                            "TSNE2","ISO1","ISO2"])
    pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(15,10), 
                               diagonal='kde', color=colors)
    plt.savefig("DataVisualization/ScatterMatrix.png", dpi=pix)
    plt.show()
    plt.close()


def parallel_coords_plot(t_data_aux):
    columns=["PCA2","PCA3","TSNE1",
             "TSNE2","ISO1","ISO2","labels"]
    '''columns=["P1","P2","P3","P4","P5","P6",
         "T1","T2","T3","T4","T5","T6",
         "I1","I2","I3","I4","I5","I6","labels"]'''
    df = pd.DataFrame(t_data_aux, columns=columns)
    pd.plotting.parallel_coordinates(df,"labels",alpha=0.5)
    plt.savefig("DataVisualization/ParallelCoordinates.png", dpi=pix)
    plt.show()
    plt.close()

def rad_viz_plot(t_data_aux):
    columns=["PCA2","PCA3","TSNE1",
             "TSNE2","ISO1","ISO2","labels"]
    df = pd.DataFrame(t_data_aux, columns=columns)
    pd.plotting.radviz(df,"labels",alpha=0.5)
    plt.savefig("DataVisualization/RadViz.png", dpi=pix)
    plt.show()
    plt.close()
