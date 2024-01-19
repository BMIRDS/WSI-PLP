conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pandas
pip install scikit-learn
pip install tqdm
pip install large-image[all] --find-links https://girder.github.io/large_image_wheels
pip install lifelines

python setup.py develop