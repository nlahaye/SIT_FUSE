import numpy as np
import ot
from joblib import load, dump

from zonal_histogram import PolygonAreaKNNGraph
 
knn_graph_fn = "/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP/PV_PANEL_base_cluster_polygon_knn_graphs.pkl"
knn_graph_fn_sf = "/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP/PV_PANEL_SF_base_cluster_polygon_knn_graphs.pkl"
  

knn_graphs = load(knn_graph_fn)
knn_graphs_sf = load(knn_graph_fn_sf)

distances = []
#thresh = 2212.062551794571
feat_thresh = 0.15551911376800703 #0.23089447818468375
feat_var = 0.02930235333289976 #0.03136828958671306

sample_thresh = 
sample_var = 

good_inds = []

for i in range(len(knn_graphs_sf)):
    for j in range(len(knn_graphs)):
 
        if ((knn_graphs_sf[i].knn_adj_final.shape[0] < knn_graphs[j].knn_adj_final.shape[0]) and (knn_graphs_sf[i].knn_adj_final.shape[0] / float(knn_graphs[j].knn_adj_final.shape[0])) < 0.7) or ((knn_graphs[j].knn_adj_final.shape[0] < knn_graphs_sf[i].knn_adj_final.shape[0]) and (float(knn_graphs[j].knn_adj_final.shape[0]) / knn_graphs_sf[i].knn_adj_final.shape[0])) < 0.7:
            continue
     


        print(i,j, len(knn_graphs_sf), len(knn_graphs),  knn_graphs_sf[i].knn_adj_final.shape, knn_graphs[j].knn_adj_final.shape)
 
        C1 = knn_graphs_sf[i].knn_adj_final
        C2 = knn_graphs[j].knn_adj_final
  
        M = np.zeros((C1.shape[0], C2.shape[0]))

        p = knn_graphs_sf[i].values_final
        q = knn_graphs[j].values_final 

        alpha = 1e-3

        distance = ot.gromov.fused_unbalanced_across_spaces_divergence(knn_graphs[i].knn_values,  knn_graphs[j].knn_values, alpha=alpha, verbose=False, log=False)
        feat_norm = np.linalg.norm(distance[1], ord='fro')
        sample_norm = np.linalg.norm(distance[0], ord='fro')
  
        #distance = ot.gromov.gromov_wasserstein(C1, C2, p, q, "kl_loss", verbose=True, log=False)

        #distance = ot.fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun="kl_loss", alpha=alpha, verbose=False, log=False)
  
        if  feat_norm >= (feat_thresh - (feat_var*2)) and sasmple_norm >= (sample_thresh - (sample_var*2)):
            good_inds.append(i)
            distances.append(distance)
            break


print(distances)
print(good_inds)





