# build.sh
# install plugins into user library
__conda_setup="$('/cpfs01/shared/pjlab-lingjun-landmarks/env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs01/shared/pjlab-lingjun-landmarks/env/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/cpfs01/shared/pjlab-lingjun-landmarks/env/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs01/shared/pjlab-lingjun-landmarks/env/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate /cpfs01/user/xingjingze/condaenv
cd plugin
rm -r build
pip install -v -e .
