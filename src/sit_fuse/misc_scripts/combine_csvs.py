
import pandas as pd 



files  = [
#"/Users/nlahaye/Downloads/HABs-NewportBeachPier_9f01_921a_45e9.csv",
#"/Users/nlahaye/Downloads/HABs-SantaMonicaPier_1ea2_e5b2_a9df.csv",
#"/Users/nlahaye/Downloads/HABs-ScrippsPier_e16e_5217_922f.csv",
#"/Users/nlahaye/Downloads/HABs-StearnsWharf_7ab9_af94_8796.csv"
"/Users/nlahaye/Downloads/Recent_Harmful_Algal_Bloom_HAB_Events.csv"
]

df = pd.concat( 
    map(pd.read_csv, files), ignore_index=True) 
print(df)
df.rename(columns={"LONGITUDE":"Longitude", "LATITUDE":"Latitude", "SAMPLE_DATE":"Sample Date", "COUNT_":"Karenia brevis abundance (cells/L)", "DEPTH": "depth"}, inplace=True)
print(df) 
df.drop(df[df.depth < 1.0].index, inplace=True) 
 
df.rename(columns={"depth": "Sample Depth"}, inplace=True)
df["Datetime"] = df["Sample Depth"]

df.to_csv("/Users/nlahaye/Downloads/New_FL_HAB_Karenia_Brevis.csv")




