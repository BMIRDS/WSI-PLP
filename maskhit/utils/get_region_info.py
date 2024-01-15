import pandas as pd
import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Get Region Info')

parser.add_argument('--meta-svs',
                    type=str,
                    help='svs meta file')
parser.add_argument('--magnification',
                    type=int,
                    default=20,
                    help='magnification level')
parser.add_argument('--patch-size',
                    type=int,
                    default=224,
                    help='patch size')
parser.add_argument('--region-size',
                    type=int,
                    default=2240,
                    help='region size')
parser.add_argument('--sampling-threshold',
                    type=int,
                    default=10,
                    help='sampling threshold')
parser.add_argument('--grid-size',
                    type=int,
                    default=10, 
                    help='grid size. Unit: patches')

args = parser.parse_args()

magnification = float(args.magnification)
patch_size = args.patch_size
region_size = args.region_size
grid_size = args.grid_size
sampling_threshold = args.sampling_threshold

length = region_size / patch_size

assert length in [4, 5, 10, 20, 50, 100]

df_svs = pd.read_pickle(args.meta_svs)

counts = []
for i, row in tqdm.tqdm(df_svs.iterrows(), total=df_svs.shape[0]):
    df_i = pd.read_pickle(
        f"data/{row['cancer']}/{row['id_svs']}/mag_{magnification}-size_{patch_size}/meta.pickle"
    )
    n_regions = df_i.loc[
        (df_i.pos_x % grid_size == 0) & (df_i.pos_y % grid_size == 0) &
        (df_i[f"counts_{int(length)}"] > sampling_threshold)].shape[0]
    counts.append(n_regions)

print("Average number of regions: ", np.mean(counts))

print(
    "Region counts percentiles:\n\tMin:\t{0}\n\t50.0%:\t{1}\n\t80.0%:\t{2}\n\t95.0%:\t{3}\n\t99.0%:\t{4}\n\tMax:\t{5}"
    .format(
        *tuple(np.quantile(counts, q=[0.00, 0.5, 0.80, 0.95, 0.99, 1.00]))))
