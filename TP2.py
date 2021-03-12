import numpy as np
import tp2_aux as aux
import aux_functions


# ------ Get Labels and images matrix --------

idx_labels = np.loadtxt("labels.txt", delimiter=",")
labels = idx_labels[:,1]
image_mat = aux.images_as_matrix()


# ----- Feature Extraction ----------

#t_data = aux_functions.feature_extraction(image_mat, "all")
t_data = np.loadtxt("feature_extraction.txt", delimiter=" ")
t_data = aux_functions.standardize(t_data)
t_biologists_data = t_data[labels>0,:]

# ------ Pre-feature Selection --------

f, selected_data, feats = aux_functions.feature_selection(t_biologists_data, 
                                                labels[labels>0], 6)
# ------- Data Visualization ----------

# color labels for better visualization of the labelled sample
colors = []
for i in labels:
    if i == 1:
        colors.append("blue")
    elif i == 2:
        colors.append("red")
    elif i == 3:
        colors.append("green")
    else:
        colors.append("white")

aux_functions.scatter_plot(t_data, feats, colors)

t_data_aux = np.zeros((t_biologists_data.shape[0], len(feats)+1))
t_data_aux[:,:len(feats)] = t_biologists_data[:,feats]
t_data_aux[:,len(feats)] = labels[labels>0]

aux_functions.parallel_coords_plot(t_data_aux)

aux_functions.rad_viz_plot(t_data_aux)


# ------- K-MEANS ----------

for i in range(2,6):
    t,_,feats = aux_functions.feature_selection(t_biologists_data,
                                                labels[labels>0],i)
    t_data_comb = t_data[:,feats]
    aux_functions.plot_kmeans_metrics(t_data_comb, labels, feats)


## -------- DBSCAN ---------
# Manual Testing
t, _, feats = aux_functions.feature_selection(t_biologists_data,
                                         labels[labels>0],3)
t_data_comb = t_data[:,feats]
aux_functions.plot_elbow(t_data_comb, feats)

aux_functions.plot_dbscan_metrics(t_data_comb, labels, feats, 
                                0.4, 0.15)

## -------- Agglomerative ---------

for i in range(2,6):
    t,_,feats = aux_functions.feature_selection(t_biologists_data,
                                                labels[labels>0],i)
    t_data_comb = t_data[:,feats]
    aux_functions.plot_agglomerative_metrics(t_data_comb, labels, feats)



# --------- Bissecting KMeans ----------

for i in range(2,6):
    t,_,feats = aux_functions.feature_selection(t_biologists_data,
                                                labels[labels>0],i)
    t_data_comb = t_data[:,feats]
    aux_functions.plot_bissecting_kmeans_metrics(t_data_comb, labels, feats)


# ------- CLUSTER REPORTS -----------

#KMeans

feats = [1,2,12]
t_data_comb = t_data[:,feats]
k_means_labels = aux_functions.k_means_clustering(8,t_data_comb)
aux.report_clusters(idx_labels[:,0],k_means_labels,"test_kmeans.html")


#DBSCAN

feats = [1,12,13]
t_data_comb = t_data[:,feats]
db_labels = aux_functions.dbscan_clustering(0.37,t_data_comb)
aux.report_clusters(idx_labels[:,0],db_labels,"test_db.html")

# Agglomerative

feats = [1,2,12]
t_data_comb = t_data[:,feats]
agg_labels = aux_functions.agglomerative_clustering(9,t_data_comb)
aux.report_clusters(idx_labels[:,0],agg_labels,"test_agg.html")


# Bissecting KMeans


feats = [1,2,6,12,13]
t_data_comb = t_data[:,feats]
result = aux_functions.bissecting_kmeans_clustering(t_data_comb, feats, 10)
aux.report_clusters_hierarchical(idx_labels[:,0],result,"test_bisK.html")
