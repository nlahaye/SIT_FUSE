import pandas as pd
 
init_fname = "/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/InSitu/W_FL_Karenia_brevis_abundance_2002-2021.04.05.xlsx"
 
print(init_fname)
insitu_df = pd.read_excel(init_fname)
# Format Datetime Stamp
insitu_df['Datetime'] = pd.to_datetime(insitu_df['Sample Date'])
insitu_df.set_index('Datetime')
 
insitu_df = insitu_df[insitu_df['Sample Depth (m)'] <= 1.0]

test = insitu_df.groupby(['Sample Date','Latitude','Longitude'])['Karenia brevis abundance (cells/L)'].mean().reset_index(name='Karenia brevis abundance (cells/L)')
print(test)

insitu_df.to_csv('/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/InSitu/W_FL_Karenia_brevis_abundance_2002-2021.04.05.updated.csv') 



