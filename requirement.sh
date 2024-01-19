# create a virtual environment
# conda create -n hidp
# conda activate hidp

# install pytorch :https://pytorch.org
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install openslide
pip install openslide-python

# install additional packages
pip install pandarallel pandas scikit-image scikit-learn einops tqdm lifelines pyyaml
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install opencv-python
