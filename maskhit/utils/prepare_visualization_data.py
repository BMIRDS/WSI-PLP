import pandas as pd

region_size = 2240
study = 'IBD_PROJECT' # change to name of disease/cancer

# meta information about dataset e.g. case_number, diagnosis, etc.
df_meta = pd.read_pickle('meta/ibd_project_meta.pickle')

# contains svs_path, svs_id, valid_counts
df_svs = pd.read_pickle('meta/ibd_project_svs.pickle')

# slides to select for visualization, if empty all slides selected
sel_ids = [
    '10SP1919976 A1-1_SS12253_183313', # HCTP for Cr
    '10SP1919976 B1-1_SS12253_183037', # HCTP for Cr
    '10SP1904090 C1-1_SS12253_011833', # HCFP for UC
    '10SP1906551 B1-1_SS12253_012834', # HCFP for UC
    '10SP1903499 F1-1_SS12253_220812', # HCTP for Cr
    '10SP1911646 J1-1_SS12253_204104' # HCTP for Cr
]

df_svs = df_svs.loc[df_svs.id_svs.isin(sel_ids)]
df_meta = df_meta.loc[df_meta.case_number.isin(df_svs.case_number)]

res = []
for i, row in df_svs.iterrows():
    print(row['id_svs'])
    df_i = pd.read_pickle(
        f"data/{study}/{row['id_svs']}/mag_20-size_224/meta.pickle")
    
    # counts_x where x is dependent on region_size
    if region_size == 2240:
        df_i = df_i.loc[df_i.counts_10 > 0, ['pos']]
    else:
        df_i = df_i.loc[df_i.counts_20 > 0, ['pos']]

    df_i['case_number'] = row['case_number']
    df_i['id_svs'] = row['id_svs']
    df_i['svs_path'] = row['svs_path']
    res.append(df_i)

df_locs = pd.concat(res)
df_locs['slide_type'] = 'ffpe'
df_locs['cancer'] = study

df_meta.merge(df_locs[['case_number','svs_path','id_svs']].drop_duplicates('id_svs'), on='case_number')

df_locs.to_pickle(f'meta/vis_{study.lower()}_locs-split.pickle') # stores location of patches extracted from specified WSIs 
df_meta.to_pickle(f'meta/vis_{study.lower()}_meta-split.pickle') # stores meta information of WSIs
