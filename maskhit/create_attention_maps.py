import torch
import numpy
import pandas as pd
import pickle
import tqdm
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from scipy.ndimage import gaussian_filter
from cv2 import resize
from PIL import Image
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np
from skimage.util.shape import view_as_windows
import cv2
from vis.wsi import WSI

def restore_agg_attn_map(row):
    """
    Parameters:
    - row (pd.Series): expected to have following keys: model_name, study, and epoch

    Returns:
    - attn_maps (dict): Dictionary containing attention maps, loaded from a pickle file.

    Note: The function assumes the existence of a features_agg directory with a specified directoru structure
    """

    model_name, study, epoch = row['model_name'], row['study'], row['epoch']
    try:
        # with study name as part of path to attention map file
        with open(f"features_agg/{model_name}/None-None/{study}/{epoch}/attn_map.pickle", "rb") as f:
            attn_maps = pickle.load(f)
    except FileNotFoundError:
        try:
            # without study name as part of path to attention map file
            with open(f"features_agg/{model_name}/None-None/{epoch}/attn_map.pickle", "rb") as f:
                attn_maps = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error: {e}")
    return attn_maps

def get_diff_map(meta_ckp, id_ft, id_pt, id_svs):
    """
    Compute and visualize the difference between attention maps for fine-tuned (ft) and pre-trained (pt) models.

    Parameters:
    - meta_ckp (pd.DataFrame): df containing meta info about models
    - id_ft (int): id # for row in meta_ckp that represents the fine-tuned model
    - id_pt (int or None): id #  for row in meta_ckp that represents the pre-trained model
    - id_svs (str): Identifier for the WSI to be analyzed.

    Returns:
    - attn_maps_diff: Difference between the fine-tuned and pre-trained model attention maps.
    - name (str): Location where the plot was saved
    - study (str): Study name related to the fine-tuned model.
    - attn_maps_ft: Attention map for the fine-tuned model on the specified WSI.
    - attn_maps_pt: Attention map for the pre-trained model on the specified WSI.

    Note: Creates a subplot of the attention maps for the fine-tuned and pre-trained model. Also includes
    the map of the difference between the two. Saves these plots in the output directory.
    """
    # extracting row corresponding to fine-tuned model
    row_ft = meta_ckp.iloc[id_ft]
    # storing the attention map from fine-tuned model as a dict
    attn_maps_ft = restore_agg_attn_map(row_ft)

    if id_pt is None:
        name = f"ft{row_ft['model_name']}-pt-None"
        return attn_maps_ft[id_svs], name, row_ft['study']
    
    # extracting row corresponding to pre-trained model
    row_pt = meta_ckp.iloc[id_pt]
    # storing the attention map from pre-trained model as a dict
    attn_maps_pt = restore_agg_attn_map(row_pt)
    
    name = f"ft{row_ft['model_name']}-pt{row_pt['model_name']}"
    fig, axes = plt.subplots(1,3, figsize=(10,15))
    attn_maps_diff = attn_maps_ft[id_svs] - attn_maps_pt[id_svs]
    
    # Fine-tuned Attention Map
    ft_map = axes[0].imshow(attn_maps_ft[id_svs], cmap='coolwarm')
    axes[0].axis('off')
    axes[0].set_title('Fine-tuned Attention Map')
    fig.colorbar(ft_map, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

    # Pre-trained Attention Map
    pt_map = axes[1].imshow(attn_maps_pt[id_svs], cmap='coolwarm')
    axes[1].axis('off')
    axes[1].set_title('Pre-trained Attention Map')
    fig.colorbar(pt_map, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Difference Map
    diff_map = axes[2].imshow(attn_maps_diff, cmap='coolwarm')
    axes[2].axis('off')
    axes[2].set_title('Difference Map')
    fig.colorbar(diff_map, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    # Save the figure
    save_path = f"output/{name}"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Attention Map Subplots saved at {save_path}")
    return attn_maps_diff, name, row_ft['study'], attn_maps_ft[id_svs], attn_maps_pt[id_svs]

def get_correct_thumbnail(meta_ckp, id_row, id_svs):
    """
    Returns the appropriate thumbnail image and WSI object
    
    Parameters:
    - meta_ckp (pd.DataFrame): df containing meta info about models
    - id_row (int): id for row in the `meta_ckp` df.
    - id_svs (int or str): Id for SVS slide.
    
    Returns:
    - wsi (WSI object): WSI object with path to svs file
    - thumbnail (numpy.ndarray): Thumbnail version of WSI
    """
    row = meta_ckp.iloc[id_row]
    try:
        df_meta = pd.read_pickle(f"features/{row['model_name']}/None-None/{row['study']}/{row['epoch']}/meta.pickle")
    except FileNotFoundError:
        df_meta = pd.read_pickle(f"features/{row['model_name']}/None-None/{row['epoch']}/meta.pickle")

    # selects the first row of a patch selected from a specified svs 
    row_svs = df_meta.loc[df_meta.id_svs_num == int(id_svs)].iloc[0]

    svs_path = row_svs['svs_path']
    old_base_path = "/pool2/data/WSI_IBD/svs_2019/"
    new_base_path = "/home/datasets/WSI_IBD/svs_2019/"
    svs_path = svs_path.replace(old_base_path, new_base_path)
    wsi = WSI(svs_path) # WSI object created with path
    mag_mask = 1.25
    thumbnail = wsi.downsample(mag_mask)
    return wsi, thumbnail

def crop_prop_img(img):
    size_y, size_x, _ = img.shape
    # changed from 10 to 20 (could be related to magnification)
    down_scale = 20/mag_mask/224
    max_x, max_y = int(size_x*down_scale), int(size_y*down_scale)
    new_y, new_x = int(max_x/down_scale), int(max_y/down_scale)
    img = img[:new_x, :new_y, :]
    return img, (max_x, max_y), (new_x, new_y)

def is_purple_dot(r, g, b):
    """
    Determines if a pixel is of a purple
    """
    rb_avg = (r + b) / 2
    if r > g - 20 and b > g - 20 and rb_avg > g + 10:
        return True
    return False

def is_purple(crop):
    """
    Determines if there's any purple pixel within a 2x2 cropped image section.
    """
    crop = crop.reshape(2,2,3)
    for x in range(crop.shape[0]):
        for y in range(crop.shape[1]):
            r = crop[x, y, 0]
            g = crop[x, y, 1]
            b = crop[x, y, 2]
            if is_purple_dot(r, g, b):
                return True
    return False

def filter_purple(img):
    """
    Creates a binary mask for img where purple pixels are 1 and non-purple pixels are 0
    """
    h,w,d = img.shape
    step = 2
    img_padding = np.zeros((h+step-1,w+step-1,d))
    img_padding[:h,:w,:d] = img
    img_scaled = view_as_windows(img_padding, (step,step,3), 1)
    return np.apply_along_axis(is_purple, -1, img_scaled.reshape(h,w,-1)).astype(int)

def plot_local_attention(wsi, attn_map3, vmin, vmax, pos_x, pos_y, region_size, magnification, annotation=None, save_fig=False, save_dir=""):
    """
    Plots local attention map of a specified region
    
    Parameters:
    - wsi: WSI object
    - attn_map3 (numpy.ndarray): 2D array representing the attention map.
    - vmin, vmax
    - pos_x (int): X-coordinate of the top-left corner of region to be visualized.
    - pos_y (int): Y-coordinate of the top-left corner of region to be visualized.
    - region_size (int): Size of the region to be visualized.
    - magnification (float): magnification of WSI.
    - annotation (list): List of tuples (x, y, size). Default is None
    - save_fig (bool): If True, save the figures as SVG
    - save_dir (str): Directory path to save the figures

    Returns: None
    """

    # selecting a specific potion of the attention map
    attn_clip = attn_map3[(max(0, pos_y)):(pos_y+region_size), (max(0,pos_x)):(pos_x+region_size)]
    region_name = f"r_{region_size}-m_{str(magnification)}-{pos_x}_{pos_y}"
    tile = wsi.get_region(pos_x, pos_y, region_size*magnification/1.25, magnification, 1.25)
    
    # plotting selected tile extracted based on pos_x and pos_y
    fig = plt.figure(figsize=(8,8))
    plt.imshow(tile)
    plt.axis('off')
    if save_fig:
        plt.savefig(save_dir + f"{region_name}-tile.svg",bbox_inches='tight',pad_inches = 0)
        print(f"Saved figure at {save_dir} directory")
        plt.close()
    else:
        plt.show()

    # plotting attn_clip
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_axes((0,0,1,1), label='wsi')
    ax2 = fig.add_axes((0,0,1,1), label='attn')
    ax1.imshow(attn_clip, vmin=vmin, vmax=vmax, alpha=1.0, cmap='coolwarm')
    ax1.axis('off')        
    ax2.imshow(tile, alpha=0.5)
    ax2.axis('off')

    # adding annotations to the image
    if annotation is None:
        pass
    else:
        for pos in annotation:
            pos_x, pos_y, region_size = pos
            rect = mp.Rectangle((pos_x,pos_y), region_size, region_size, linewidth=2, edgecolor='black', facecolor='none')
            ax2.add_patch(rect)     
    if save_fig:
        plt.savefig(save_dir + f"{region_name}-overlay.svg",bbox_inches='tight',pad_inches = 0)
        print(f"Saved figure at {save_dir} directory")
        plt.close()
    else:
        plt.show()

def get_local_attention_overlay(save_dir, attn_map3, wsi, vmin, vmax, sel_region, sel_patch, save_fig=False):
    """
    Seems to repetitively plot local attention maps based on specified patch locations

    Parameters:
    - save_dir (str): Directory path where the figures will be saved, if `save_fig` is True.
    - attn_map3 (numpy.ndarray): 2D array representing the attention map.
    - wsi: WSI object
    - vmin, vmax
    - sel_region (tuple): tuple with following structure (pos_x, pos_y, region_size, magnification).
    - sel_patch (list): List of tuples with format: (pos_x, pos_y, region_size, magnification).
    - save_fig (bool): If True, save the figures in the specified `save_dir`. Default is False.
    """     
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(attn_map3)
    for sel_i in [sel_region]+sel_patch:
        pos_x, pos_y, region_size, _ = sel_i
        rect = mp.Rectangle((pos_x,pos_y), region_size, region_size, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
    pos_x, pos_y, region_size, magnification = sel_region
    annotation = []
    for sel_i in sel_patch:
        pos_xp, pos_yp, region_size_p, _ = sel_i
        annotation.append([pos_xp-pos_x, pos_yp-pos_y,region_size_p])
    plot_local_attention(wsi, attn_map3, vmin, vmax, pos_x, pos_y, region_size, magnification, annotation, save_fig=save_fig, save_dir=save_dir)
    for i, sel_i in enumerate(sel_patch):
        pos_xp, pos_yp, region_size, magnification = sel_i
        plot_local_attention(wsi, attn_map3, vmin, vmax, pos_xp, pos_yp, region_size, magnification, save_fig=save_fig, save_dir=save_dir)

def complete_pipeline(svs_id, ft_id, pt_id, save_fig=True, cl=0.2, ch=0.8, sel_region=None, sel_patch=None, alpha=0.3):
    """
    Complete pipeline for visualizing the attention overlay on whole slide images, based on the attention map differences.

    Parameters:
    - svs_id: Id for the whole slide image (WSI).
    - ft_id: Id for the fine-tuned model.
    - pt_id: Id for the pre-trained model.
    - save_fig (bool): Flag indicating whether to save figures. Default is True.
    - cl, ch
    - sel_region (tuple): tuple of the form (pos_x, pos_y, region_size, magnification). Default is None.
    - sel_patch (list, optional): List of tuples of the form (pos_x, pos_y, region_size, magnification). Default is None.
    - alpha (float, optional): Transparency level for overlays on top of thumbnails. Default is 0.3.

    Returns:
    - attn_map3 (numpy.ndarray): 2D attention map array for the selected WSI.
    - thumbnail2
    - wsi: WSI object.
    """
    # attn_map refers to the difference attention map
    attn_map, map_name, study, _, _ = get_diff_map(df_meta_ckp, ft_id, pt_id, svs_id)
    wsi, thumbnail = get_correct_thumbnail(df_meta_ckp, ft_id, svs_id)

    thumbnail2, (max_x, max_y), (new_x, new_y) = crop_prop_img(thumbnail)

    attn_map[np.isnan(attn_map)] == 0
    attn_map = resize(attn_map, (new_y, new_x), interpolation=cv2.INTER_LINEAR)

    if wsi.svs_path not in masks.keys():
        mask = filter_purple(thumbnail2)
        masks[wsi.svs_path] = remove_small_holes(mask==1,400)
    mask = masks[wsi.svs_path]

    id_patient = wsi.svs_path.split('/')[-1][:12]
    save_dir = f"output/visualization/{study}/{id_patient}/{map_name}/"
    os.makedirs(save_dir, exist_ok=True)

    attn_map2 = attn_map.copy()
    attn_map2[np.isnan(attn_map2)] = 0
    attn_map3 = mask*attn_map2
    attn_map3[attn_map3 == 0] = np.nan 

    vmin, vmax = np.nanquantile(attn_map3, cl),np.nanquantile(attn_map3, ch)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_axes((0,0,1,1), label='wsi')
    ax2 = fig.add_axes((0,0,1,1), label='attn')  

    am = attn_map3.copy()
    am[am > vmax] = vmax
    am[am < vmin] = vmin
    # plotting the attention map of WSI
    im = ax1.imshow(attn_map3, alpha=1.0, vmin=vmin, vmax=vmax, cmap='coolwarm')
    ax1.axis('off')
    # showing original thumbnail
    ax2.imshow(thumbnail2, alpha=0.5)
    ax2.axis('off')

    if sel_region is not None:
        pos_x, pos_y, region_size,_ = sel_region
        rect = mp.Rectangle((pos_x,pos_y), region_size, region_size, linewidth=2, edgecolor='black', facecolor='none')
        ax2.add_patch(rect)
    
    if save_fig:
        plt.savefig(save_dir+'overview-combined.svg',bbox_inches='tight',pad_inches = 0)
        plt.close()
    else:
        plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.imshow(thumbnail2)
    plt.axis('off')
    if save_fig:
        plt.savefig(save_dir+'overview-thumbnail.svg',bbox_inches='tight',pad_inches = 0)
        plt.close()
    else:
        plt.show()

    fig = plt.figure(figsize=(10,10))
    im = plt.imshow(attn_map3, vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.colorbar(im, orientation='vertical', shrink=0.75)  # adding colorbar
    plt.axis('off')
    if save_fig:
        plt.savefig(save_dir+'overview-heatmap.svg',bbox_inches='tight',pad_inches = 0)
        plt.close()
    else:
        plt.show()
    
    if sel_region is not None and sel_patch is not None:
        get_local_attention_overlay(save_dir, thumbnail, attn_map3, wsi, vmin, vmax, sel_region=sel_region, sel_patch=sel_patch, save_fig=save_fig)

    return attn_map3, thumbnail2, wsi

def get_attention_map(attn_map, patch_i = -1):
    # changed from .view(20, 20) to .view(10, 10) to accomodate for 100 input size
    mask = attn_map[patch_i+1,1:].view(10,10)
    n = (mask > 0).sum()
#     return torch.log(mask*n)
    return mask*n

def compress_global_attention(model_name, study, epoch, svs_dir, layers_i = 'None', head_i = 'None', plot_result=False):
    '''
    Saves plots of attention maps for a given model on a specified SVS WSI
    '''
    df = pd.read_pickle(f"features/{model_name}/{layer_i}-{head_i}/{epoch}/meta.pickle")
    svs_ids = df.id_svs.unique().tolist()
    df.drop_duplicates('id_svs', inplace=True)

    attn_maps = {}
    cnts_maps = {}
    dims_maps = {}
    thumbnails = {}
    pattern = f"features/{model_name}/{layer_i}-{head_i}/{epoch}/*.pickle"
    files = glob.glob(pattern)
    for fname in tqdm.tqdm(files):
        if os.path.basename(fname) == 'meta.pickle':
            continue
        with open(fname, 'rb') as f:
            batch_inputs = pickle.load(f)
        n_samples = batch_inputs['ids'].size(0)
        for i in range(n_samples):
            pos_x, pos_y = batch_inputs['pos_tile'][i].squeeze().tolist()
            svs_id = str(batch_inputs['ids'][i].item())
            if svs_id not in attn_maps.keys():
                svs_path = df.loc[df.id_svs_num == int(svs_id), 'svs_path'].item()
                old_base_path = "/pool2/data/WSI_IBD/svs_2019/"
                new_base_path = "/home/datasets/WSI_IBD/svs_2019/"
                svs_path = svs_path.replace(old_base_path, new_base_path)
                wsi = WSI(svs_path)
                mag_mask = 1.25
                thumbnail = wsi.downsample(mag_mask)
                thumbnail, (max_x, max_y), (new_x, new_y) = crop_prop_img(thumbnail)
                # changed from 20 to 10
                attn_maps[svs_id] = np.zeros((max_x+10, max_y+10))
                cnts_maps[svs_id] = np.zeros((max_x+10, max_y+10))                    
                dims_maps[svs_id] = (max_x, max_y)
                thumbnails[svs_id] = thumbnail
            attn_map_i = get_attention_map(batch_inputs['attn'][i], -1).numpy()
            # changed from 20 to 10
            attn_maps[svs_id][pos_x:(pos_x+10),pos_y:(pos_y+10)] += attn_map_i
            cnts_maps[svs_id][pos_x:(pos_x+10),pos_y:(pos_y+10)] += attn_map_i != 0

    for svs_id in list(attn_maps.keys()):
        max_x, max_y = dims_maps[svs_id]
        attn_maps[svs_id] = attn_maps[svs_id]/cnts_maps[svs_id]
        attn_maps[svs_id] = attn_maps[svs_id][:max_x, :max_y]
        attn_maps[svs_id] = attn_maps[svs_id].swapaxes(1,0)
        
        if plot_result:
            fig = plt.figure(figsize=(8,8))
            axis_1 = fig.add_axes([0,0,1,1], label='1')
            axis_2 = fig.add_axes([0,0,1,1], label='2')
            axis_1.imshow(thumbnails[svs_id])
            axis_1.axis('off')

            axis_2.imshow(attn_maps[svs_id], alpha=0.4, cmap='viridis')
            axis_2.axis('off')

            save_path = f"{model_name} {svs_id}.png"
            print(f"Saved attention maps at {save_path}")
            plt.savefig(save_path)
            plt.close(fig)

    save_loc = f"features_agg/{model_name}/{layer_i}-{head_i}/{epoch}/"
    os.makedirs(save_loc, exist_ok=True)

    with open(save_loc+'attn_map.pickle', 'wb') as f:
        pickle.dump(attn_maps, f)

if __name__ == '__main__':
    meta_dict = [
        {
            "order": 0,
            "name": "TCGA_pretrained",
            "study": 'IBD_PROJECT',
            "model_name": "-pretrained_20x_448_resnet34",
            "epoch": "0500",
            "svs_dir": "svs_2019"
        },

        {
            "order": 1,
            "name": "IBD_classification",
            "study": 'IBD_PROJECT',
            "model_name": "-2023_5_30-0",
            "epoch": "0006",
            "svs_dir": "svs_2019"
        }
    ]

    df_meta_ckp = pd.DataFrame(meta_dict)
    df_meta_ckp = df_meta_ckp.sort_values('order').reset_index(drop=True)
    print(df_meta_ckp)

    layer_i = head_i = 'None'
    for i, row in df_meta_ckp.iterrows():
        study = row['study']
        model_name = row['model_name']
        epoch = row['epoch']
        svs_dir = row['svs_dir']

        try:
            df_meta = pd.read_pickle(f'features/{model_name}/{layer_i}-{head_i}/{epoch}/meta.pickle')
        except Exception as e:
            print(e)
            continue
        
        if 'svs_path' in df_meta.columns:
            continue
        
        res = []
        for i_i, row_i in df_meta.drop_duplicates('id_svs_num').iterrows():
            svs_id = row_i['id_svs']
            file_id = df_meta.loc[df_meta.id_svs == svs_id].id_svs.tolist()[0]
            svs_path = glob.glob(f"/pool2/data/WSI_IBD/{svs_dir}/{file_id}.svs")[0]
        
        svs_path_mapping = pd.DataFrame(res, columns=['id_svs','svs_path'])
        df_meta = df_meta.merge(svs_path_mapping, on='id_svs')
        df_meta.to_pickle(f'features/{model_name}/{layer_i}-{head_i}/{study}/{epoch}/meta.pickle')
        print(f'features/{model_name}/{layer_i}-{head_i}/{epoch}/meta.pickle')

    mag_mask = 1.25

    # create masks if not exist
    try:
        print(masks.keys())
    except:
        masks = {}
    
    for i, row in df_meta_ckp.iterrows():
        print(i, row)
        study = row['study']
        model_name = row['model_name']
        epoch = row['epoch']
        svs_dir = row['svs_dir']
        print('<>'*30)
        print(model_name, study, epoch, svs_dir)
        compress_global_attention(model_name, study, epoch, svs_dir, plot_result=True)
    
    complete_pipeline(svs_id='1', pt_id=0, ft_id=1)
