# install openslide
pip install openslide-python

# install additional packages
pip install pandarallel pandas scikit-image scikit-learn einops tqdm lifelines pyyaml
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install opencv-python
# When you see: AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
# Reinstall older working version: pip install opencv-python==4.8.0.74
