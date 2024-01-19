import torch
import numpy as np
import math
import pandas as pd


class WsiTileSampler:

    def __init__(self,
                 data,
                 sample_all=True,
                 mode='train',
                 num_patches=100,
                 args=None):
        self.df = data
        self.df['fid'] = self.df.index.astype(int)
        self.sample_all = sample_all
        self.mode = mode
        self.args = args
        self.num_patches = num_patches
        self.grid_size = args.grid_size

    def sample_globally(self, n):
        # sample from all the valid locations over the entire WSI
        df_valid = self.df.loc[self.df.valid == 1]
        n_valid = df_valid.shape[0]

        assert n > 0, "number of patches to be sampled has to be larger than 0"

        n_patches_valid = min(n_valid, n)

        df_sample = df_valid.sample(n, replace=True)
        pct_valid = n_patches_valid / n

        loc_tile = [0, 0]
        locs_global = locs_local = df_sample.pos.tolist()
        fids = df_sample.fid.tolist()

        output = {
            'loc_tile': loc_tile,
            'locs_local': locs_local,
            'locs_global': locs_global,
            'fids': fids,
            'pct_valid': pct_valid,
        }
        return output

    def sample_locally(self, n, region_length, loc_tile):
        # if loc_tile is None:
        #     n = region_length * region_length if self.sample_all else n
        #     return torch.zeros(n, 512), [-1,
        #                                  -1], [[0, 0] for _ in range(n)], None
        x, y = loc_tile
        x_range = range(x, x + region_length)
        y_range = range(y, y + region_length)
        df_sel = self.df.loc[self.df.pos_x.isin(x_range)
                             & self.df.pos_y.isin(y_range)].copy()

        # get all posible positions
        full_pos = np.indices((region_length, region_length)).reshape(2, -1).swapaxes(1, 0)
        df_full = pd.DataFrame(full_pos, columns=['x', 'y'])

        # original positions
        df_sel['x'] = df_sel.pos_x - x
        df_sel['y'] = df_sel.pos_y - y

        # combine them
        df_full = df_full.merge(df_sel, on=['x', 'y'], how='left')
        df_full.valid.fillna(0, inplace=True)
        df_full.loc[df_full.valid == 0, 'fid'] = -1
        df_full.pos_x.fillna(0, inplace=True)
        df_full.pos_y.fillna(0, inplace=True)
        df_full['pos'] = df_full.apply(lambda x: [x['pos_x'], x['pos_y']],
                                       axis=1)

        fids = df_full.fid.to_numpy()
        locs_global = np.stack(df_full.pos.tolist())
        locs_local = np.indices((region_length, region_length))
        if self.mode == 'extract':
            pass
        else:
            # random rotate
            locs_local = np.rot90(locs_local, np.random.randint(4), (1, 2))

            # random vertical flip
            if np.random.rand(1) < 0.5:
                locs_local = np.flip(locs_local, axis=1)
            # random horizontal flip
            if np.random.rand(1) < 0.5:
                locs_local = np.flip(locs_local, axis=2)

        locs_local = locs_local.reshape(2, -1).swapaxes(1, 0)

        valid_locs = np.where(fids > -1)[0]
        n_valid = valid_locs.shape[0]

        invalid_locs = np.where(fids == -1)[0]

        if self.sample_all:
            pass
        elif self.num_patches < region_length * region_length:

            if self.num_patches <= n_valid:
                # select only valid patches
                _sel = np.random.choice(valid_locs,
                                        self.num_patches,
                                        replace=False)
            else:
                # select all valid patches
                # if not enough, select invalid patches
                _sel_valid = np.random.choice(valid_locs,
                                              n_valid,
                                              replace=False)
                _sel_invalid = np.random.choice(invalid_locs,
                                                self.num_patches - n_valid,
                                                replace=False)
                _sel = np.concatenate([_sel_valid, _sel_invalid])

            locs_local = locs_local[_sel]
            locs_global = locs_global[_sel]
            fids = fids[_sel]

        locs_local = locs_local.tolist()

        pct_valid = min(n_valid, self.num_patches) / self.num_patches

        output = {
            'loc_tile': loc_tile,
            'locs_local': locs_local,
            'locs_global': locs_global,
            'fids': fids,
            'pct_valid': pct_valid,
        }

        return output

    def find_eligible_offset(self, region_length):
        mc = f"counts_{region_length}"
        step = self.grid_size
        psp = step * step  # possible starting points

        dict_offsets = {}
        max_patches = 0
        while max_patches == 0:
            for _ in range(10):
                offset = np.random.choice(psp, 1).item()
                offset_x = offset // step
                offset_y = offset % step
                nonzero_regions = self.df.loc[
                    (self.df.pos_x % step == offset_x)
                    & (self.df.pos_y % step == offset_y) &
                    (self.df[mc] > 0)].shape[0]
                max_patches = self.df.loc[(self.df.pos_x % step == offset_x)
                                          & (self.df.pos_y %
                                             step == offset_y)][mc].max()
                if math.isnan(nonzero_regions):
                    nonzero_regions = 0
                if math.isnan(max_patches):
                    max_patches = 0
                dict_offsets[(offset_x, offset_y,
                              max_patches)] = nonzero_regions
            offset_x, offset_y, max_patches = max(dict_offsets,
                                                  key=dict_offsets.get)
            nonzero_regions = dict_offsets[(offset_x, offset_y, max_patches)]
            # print(offset_x, offset_y, nonzero_regions, max_patches)
            if max_patches == 0:
                print('<>' * 30)
                print("Glitch found in sampling patches! Will retry")
                print(dict_offsets)
        return offset_x, offset_y

    def sample_tiles(self,
                     region_length,
                     threshold,
                     num_regions,
                     loc=None,
                     weighted_sample=False):

        if loc is not None:
            return [loc]

        mc = f"counts_{region_length}"

        step = self.grid_size
        psp = step * step  # possible starting points

        grid_sampling_mode = num_regions > 1
        if grid_sampling_mode:
            if self.args.visualization:
                offset_x = offset_y = 0
            offset_x, offset_y = self.find_eligible_offset(region_length)
            _df = self.df.loc[(self.df.pos_x % step == offset_x)
                              & (self.df.pos_y % step == offset_y)]

        if weighted_sample:
            criterion = 1
        elif grid_sampling_mode:
            criterion = min(threshold, max(_df[mc].nlargest(20).min(), 1))
        else:
            criterion = min(threshold, self.df[mc].nlargest(20).min())

        # will not sample from the edge
        if grid_sampling_mode:
            _df = _df.loc[self.df[mc] >= criterion]
        else:
            _df = self.df.loc[(self.df[mc] >= criterion)]

        if weighted_sample:
            locs = _df.pos.sample(num_regions, weights=_df[mc],
                                  replace=True).tolist()
        elif self.args.outcome_type == 'mlm':
            locs = _df.pos.sample(num_regions, replace=True).tolist()
        else:
            num_regions_valid = _df.shape[0]
            locs = _df.pos.sample(min(num_regions, num_regions_valid),
                                  replace=False).tolist()
        return locs

    def sample(self,
               n,
               region_length,
               threshold,
               num_regions=1,
               loc=None,
               weighted_sample=False):

        if loc is not None:
            locs = [loc]
        elif region_length == 0:
            locs = [[0, 0] for _ in range(num_regions)]
        else:
            locs = self.sample_tiles(region_length, threshold, num_regions, loc,
                                     weighted_sample)

        # sample for given start locations
        loc_tiles = []
        locs_local = []
        locs_global = []
        fids = []
        pct_valid = []
        for i, loc in enumerate(locs):
            if region_length == 0:
                one_tile = self.sample_globally(n)
            else:
                one_tile = self.sample_locally(n, region_length, loc)

            loc_tiles.append(torch.tensor(one_tile['loc_tile']))
            locs_local.append(torch.tensor(one_tile['locs_local']))
            locs_global.extend(one_tile['locs_global'])
            fids.append(torch.tensor(one_tile['fids']))
            pct_valid.append(torch.tensor(one_tile['pct_valid']))

        output = {
            'loc_tiles': torch.stack(loc_tiles),
            'locs_local': torch.stack(locs_local),
            'locs_global': locs_global,
            'fids': torch.stack(fids),
            'pct_valid': torch.stack(pct_valid)
        }

        return output
