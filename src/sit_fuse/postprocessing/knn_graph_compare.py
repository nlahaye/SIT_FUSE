import numpy as np
import ot
from joblib import load, dump

from zonal_histogram import PolygonAreaKNNGraph
 
knn_graph_fn = "/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP/PV_PANEL_base_cluster_polygon_knn_graphs.pkl"
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/PV_PANEL_base_cluster_polygon_knn_graphs.pkl"
  

knn_graphs = load(knn_graph_fn)

distances = []
distances_sample = []
for i in range(len(knn_graphs)-1):
    distance_sub = []
    distance_sample_sub = []
    for j in range(1, len(knn_graphs)):

        if i == j:
            continue
 
        #if ((knn_graphs[i].knn_adj_final.shape[0] < knn_graphs[j].knn_adj_final.shape[0]) and (knn_graphs[i].knn_adj_final.shape[0] / float(knn_graphs[j].knn_adj_final.shape[0])) < 0.7) or ((knn_graphs[j].knn_adj_final.shape[0] < knn_graphs[i].knn_adj_final.shape[0]) and (float(knn_graphs[j].knn_adj_final.shape[0]) / knn_graphs[i].knn_adj_final.shape[0])) < 0.7:
        #    #continue


        #print(i,j, len(knn_graphs))
 
        C1 = knn_graphs[i].knn_adj_final.toarray()
        C2 = knn_graphs[j].knn_adj_final.toarray()
        #print(C1.shape, C2.shape, C1, C2)  

        M = np.zeros((C1.shape[0], C2.shape[0]))

        p = knn_graphs[i].values_final
        q = knn_graphs[j].values_final 

        alpha = 1e-3
 
        #distance = ot.gromov.gromov_wasserstein(C1, C2, p, q, "kl_loss", verbose=True, log=False)
 
        distance = ot.gromov.fused_unbalanced_across_spaces_divergence(knn_graphs[i].knn_values,  knn_graphs[j].knn_values, alpha=alpha, verbose=False, log=False) 
        #distance = ot.gromov.fused_unbalanced_gromov_wasserstein2(C1, C2, wx=p, wy=q, verbose=False, log=False, alpha=alpha, reg_marginals=0.1, epsilon=1, divergence="l2")
        #distance = ot.gromov.partial_gromov_wasserstein2(C1, C2, p, q, loss_fun="kl_loss", alpha=alpha, verbose=False, log=False)
        #distance = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun="kl_loss", alpha=alpha, verbose=False, log=False)
        #distance = ot.gromov.BAPG_fused_gromov_wasserstein2(M, C1, C2, p=p, q=q, loss_fun="square_loss", alpha=alpha, verbose=False, log=False)
        #distance = ot.gromov.entropic_fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun="kl_loss", alpha=alpha, verbose=False, log=False, solver="PPA") 


        feat_norm = np.linalg.norm(distance[1], ord='fro')
        sample_norm = np.linalg.norm(distance[0], ord='fro')

        #print(feat_norm, distance[1].min(), distance[1].max())
        #print(sample_norm, distance[0].min(), distance[0].max())  

        distance_sub.append(abs(feat_norm))
        distance_sample_sub.append(abs(sample_norm))
    if len(distance_sub) < 1:
        continue
    print(np.min(distance_sub), np.mean(distance_sub), np.max(distance_sub), np.std(distance_sub))
    print(np.min(distance_sample_sub), np.mean(distance_sample_sub), np.max(distance_sample_sub), np.std(distance_sample_sub))
    print(i, "\n")
    distances.extend(np.sort(distance_sub)[:int(len(distance_sub)*0.1)])
    distances_sample.extend(np.sort(distance_sample_sub)[:int(len(distance_sample_sub)*0.1)])

distances = np.array(distances)
distances_sample = np.array(distances_sample)
print(distances.min(), distances.mean(), distances.max(), distances.std())
print(distance_sample_sub.min(), distance_sample_sub.max(), distance_sample_sub.std())





