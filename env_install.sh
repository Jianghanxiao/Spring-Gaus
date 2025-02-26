conda create -y -n Spring_Gaus python=3.10
conda activate Spring_Gaus
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-12.1

# install submodules for 3D Gaussian Splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# install pytorch3d
conda install -y numpy==1.26.4
pip install iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

pip install rtree