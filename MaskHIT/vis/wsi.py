import openslide
import numpy as np
import pandas as pd
from skimage.util.shape import view_as_windows


class WSI:
    """
    using openslide
    """

    def __init__(self, svs_path):
        self.svs_path = svs_path
        self.slide = openslide.OpenSlide(svs_path)
        self.mag_ori = int(
            float(self.slide.properties.get('aperio.AppMag', 40)))

    def get_region(self, x, y, size, mag, mag_mask):
        dsf = self.mag_ori / mag
        level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])
        dsf = mag_new / mag
        dsf_mask = self.mag_ori / mag_mask
        img = self.slide.read_region((int(x * dsf_mask), int(y * dsf_mask)),
                                     level, (int(size * dsf), int(size * dsf)))
        return np.array(img.convert('RGB').resize((size, size)))

    def downsample(self, mag):
        dsf = self.mag_ori / mag
        level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])
        dsf_new = self.mag_ori / mag_new
        img = self.slide.read_region(
            (0, 0), level,
            tuple(int(x / dsf_new) for x in self.slide.dimensions))
        sizes = tuple(int(x // dsf) for x in self.slide.dimensions)
        return np.array(img.convert('RGB').resize(sizes))

    def get_best_level_for_downsample(self, factor):
        levels = [int(x) for x in self.slide.level_downsamples]

        for i, level in enumerate(levels):
            if factor == level:
                return i
            elif factor > level:
                continue
            elif factor < level:
                return max(i - 1, 0)

        return len(levels) - 1
    
    def get_correct_thumbnail(wsi, mag_mask = 1.25):
        thumbnail = wsi.downsample(mag_mask)
        return thumbnail

    def crop_prop_img(img, mag_mask = 1.25):
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
                if WSI.is_purple_dot(r, g, b):
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
        return np.apply_along_axis(WSI.is_purple, -1, img_scaled.reshape(h,w,-1)).astype(int)