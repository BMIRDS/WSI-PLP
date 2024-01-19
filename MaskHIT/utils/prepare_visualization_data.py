import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Get Region Info')

parser.add_argument('--meta-file',
                    type=str,
                    default = 'meta/ibd_project_meta.pickle',
                    help='svs meta file')
parser.add_argument('--svs-file',
                    type=str,
                    default = 'meta/ibd_project_svs.pickle',
                    help='svs meta file')
parser.add_argument('--study-name',
                    type=str,
                    default = 'IBD_PROJECT',
                    help='name of cancer or disease')
parser.add_argument('--region-size',
                    type=int,
                    default=2240,
                    help='region size')
parser.add_argument('--ids', nargs='+', help='List of IDs')


args = parser.parse_args()

study = args.study_name

# meta information about dataset e.g. case_number, diagnosis, etc.
df_meta = pd.read_pickle(args.meta_file)

# contains svs_path, svs_id, valid_counts
df_svs = pd.read_pickle(args.svs_file)

# slides to select for visualization, specified in arguments
sel_ids = args.ids

df_svs = df_svs.loc[df_svs.id_svs.isin(sel_ids)]

fn_extract_prefix = lambda x: x.split('.')[0]
ids_selected = df_meta.case_number.apply(fn_extract_prefix)
df_meta = df_meta.loc[ids_selected.isin(df_svs.id_svs)]

res = []
for i, row in df_svs.iterrows():
    print(row)
    df_i = pd.read_pickle(
        f"data/{row['study_name']}/{row['id_svs']}/mag_20.0-size_224/meta.pickle")
    
    # counts_x where x is dependent on region_size
    if args.region_size == 2240:
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

# try merging on id_patient if possible. If not try merging on case_number
try:
    df_merged = df_meta.merge(df_locs[['id_patient', 'svs_path', 'id_svs']].drop_duplicates('id_svs'), on='id_patient')
except KeyError:
    df_merged = df_meta.merge(df_locs[['case_number', 'svs_path', 'id_svs']].drop_duplicates('id_svs'), on='case_number')


df_locs.to_pickle(f'meta/vis_{study.lower()}_locs-split.pickle') # stores location of patches extracted from specified WSIs 
df_meta.to_pickle(f'meta/vis_{study.lower()}_meta-split.pickle') # stores meta information of WSIs
