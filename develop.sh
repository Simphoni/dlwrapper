# build.sh
# install plugins into user library
source ${HOME}/.bashrc
conda activate nerf
MAX_JOBS=6 python setup.py develop --prefix=${HOME}/.local