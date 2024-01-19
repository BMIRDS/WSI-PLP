import os
import large_image
import glob
import tqdm
import numpy as np
import skimage.measure
import istarmap
import imageio
from multiprocessing import Pool
import logging
import argparse

parser = argparse.ArgumentParser(description='Patch extraction')
parser.add_argument('--cancer', type=str, default='BLCA')
parser.add_argument('--num-cpus', type=int, default=10)
parser.add_argument('--magnification', type=int, default=20)
parser.add_argument('--patch-size', type=int, default=224)
parser.add_argument('--min-purple-squares', type=int, default=100)
args = parser.parse_args()

logging.basicConfig(filename='logs/patch-extraction-%s-%s-%s.log' % (args.cancer,args.magnification,args.patch_size),format='%(message)s',level=logging.DEBUG)


########################################
# help functions
########################################


def is_purple_dot(r, g, b):
	rb_avg = (r + b) / 2
	if r > g - 10 and b > g - 10 and rb_avg > g + 20:
		return True
	return False


# this is actually a better method than is whitespace, but only if your images are purple lols
def is_purple(crop):
	pooled = skimage.measure.block_reduce(
		crop, (int(crop.shape[0] / 15), int(crop.shape[1] / 15), 1),
		np.average)
	num_purple_squares = 0
	for x in range(pooled.shape[0]):
		for y in range(pooled.shape[1]):
			r = pooled[x, y, 0]
			g = pooled[x, y, 1]
			b = pooled[x, y, 2]
			if is_purple_dot(r, g, b):
				num_purple_squares += 1
	if num_purple_squares > args.min_purple_squares:
		return True
	return False

def get_patch_name(fname,x=0,y=0):
	basename = os.path.basename(fname)
	parts = basename.split('.')
	return '%s-%s_%d_%d.jpg' % (parts[0],parts[1],x,y)

def parse_patch_name(fname):
    name_raw = os.path.basename(fname).replace('.jpg','')
    sub_name,xx,yy = name_raw.split('_')
    pt1 = sub_name[:7]
    pt2 = sub_name[8:]
    return os.path.join(pt1,pt1+"-"+pt2,xx,yy+'.jpg')

def extract_patch_from_svs(fname,target_path):
	ts = large_image.getTileSource(fname)
	for tile_info in ts.tileIterator(
			scale=dict(magnification=args.magnification),
			tile_size=dict(width=args.patch_size, height=args.patch_size),
			tile_overlap=dict(x=0, y=0),
			format=large_image.tilesource.TILE_FORMAT_PIL):
		im_tile = np.array(tile_info['tile'])[:, :, :3]
		if tile_info['width'] == tile_info['height']:
			try:
				if is_purple(im_tile):
					final_path = os.path.join(target_path,parse_patch_name(get_patch_name(fname,tile_info['x'],tile_info['y'])))
					os.makedirs(os.path.dirname(final_path), exist_ok=True)
					imageio.imwrite(final_path, im_tile)
			except Exception as e:
				print(e)
	logging.info(fname)


def extract_patches_for_cancer(input_path,output_path):
	with Pool(args.num_cpus) as pool:
	    iterable = [(svs_file,output_path) for svs_file in glob.glob('%s/*/*.svs' % input_path)]
	    for _ in tqdm.tqdm(pool.istarmap(extract_patch_from_svs, iterable),total=len(iterable)):
	        pass



data_lists = {
	'LGG': "./data/WSI_TCGA/Brain"
}


if __name__ == '__main__':
	print('Working on cancer %s' % args.cancer)
	output_path = 'imgs/%s/%s_%s' % (args.cancer,args.magnification,args.patch_size)
	os.makedirs(output_path, exist_ok=True)
	extract_patches_for_cancer(data_lists[args.cancer],output_path)
