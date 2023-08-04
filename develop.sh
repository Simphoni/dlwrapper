# build.sh
# install plugins into user library
source ${HOME}/.bashrc
conda activate nerf
python setup.py develop --prefix=${HOME}/.local