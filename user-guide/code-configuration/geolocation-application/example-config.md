# Example Config

<pre class="language-yaml" data-title="gk2a_geotiff_gen.yaml" data-full-width="true"><code class="lang-yaml"><strong>gen_from_geotiffs: False
</strong>
data:
 clust_reader_type: "gtiff"
 reader_kwargs:
   no_arg: "no_arg" 
 subset_inds: [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
 create_separate: False

 
 gtiff_data: [
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/00/gk2a_ami_le1b_ir087_ea020lc_202203050000.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/00/gk2a_ami_le1b_ir087_ea020lc_202203050010.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/00/gk2a_ami_le1b_ir087_ea020lc_202203050020.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/00/gk2a_ami_le1b_ir087_ea020lc_202203050030.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/00/gk2a_ami_le1b_ir087_ea020lc_202203050050.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/01/gk2a_ami_le1b_ir087_ea020lc_202203050100.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/05/01/gk2a_ami_le1b_ir087_ea020lc_202203050110.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/00/gk2a_ami_le1b_ir087_ea020lc_202203060010.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/00/gk2a_ami_le1b_ir087_ea020lc_202203060000.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/00/gk2a_ami_le1b_ir087_ea020lc_202203060020.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/00/gk2a_ami_le1b_ir087_ea020lc_202203060030.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/00/gk2a_ami_le1b_ir087_ea020lc_202203060050.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/01/gk2a_ami_le1b_ir087_ea020lc_202203060100.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/06/01/gk2a_ami_le1b_ir087_ea020lc_202203060110.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/08/01/gk2a_ami_le1b_ir087_ea020lc_202203080110.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/08/01/gk2a_ami_le1b_ir087_ea020lc_202203080120.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/08/01/gk2a_ami_le1b_ir087_ea020lc_202203080130.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/08/01/gk2a_ami_le1b_ir087_ea020lc_202203080140.tif",
"/data/nlahaye/remoteSensing/GK2A/GK2A/L1B/EA/202203/08/01/gk2a_ami_le1b_ir087_ea020lc_202203080150.tif"
]

 
 cluster_fnames: [
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050000.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050010.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050020.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050030.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050050.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050100.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203050110.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060000.tif.clust.data_79569clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060010.tif.clust.data_79587clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060020.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060030.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060050.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060100.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203060110.tif.clust.data_79593clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203080110.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203080120.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203080130.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203080140.tif.clust.data_79590clusters.no_geo.tif",
"/data/nlahaye/output/Learnergy/GK2A_DBN/gk2a_ami_le1b_ir087_ea020lc_202203080150.tif.clust.data_79590clusters.no_geo.tif"]

<strong>context: 
</strong> apply_context: False
 clusters: [
#[7.248],
[57.312,57.303,57.357],
#[57.303],
[19.611, 19.614, 19.699, 35.861, 13.217, 19.606, 19.635, 19.035, 19.043, 19.078, 19.089, 19.085, 0.56, 0.575, 19.022, 19.028, 19.079, 0.53, 19.032, 0.503, 19.023, 7.277, 7.239]]
#75.002, 7.257,7.273,
 name: ["fire", "smoke"] #i.e. smoke
 compare_truth: False
 generate_union: False
</code></pre>
