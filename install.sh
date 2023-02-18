#!/bin/bash

sudo apt install git
sudo apt install wget

export PATH /opt/conda/bin:${PATH}
wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

conda env create -f conda-env.yaml python=3.10.9
conda activate aivtuber

cd utils

git clone https://github.com/jnordberg/tortoise-tts.git
cd tortoise-tts
python setup.py install
mv tortoise ../src/utils
cd ..
sudo rm -r tortoise-tts

git clone https://github.com/yzhou359/MakeItTalk
cd MakeItTalk
mkdir examples/dump
mkdir examples/ckpt
gdown --id 1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x -O examples/ckpt/ckpt_autovc.pth
gdown --id 1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp -O examples/ckpt/ckpt_content_branch.pth
gdown --id 1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu -O examples/ckpt/ckpt_speaker_branch.pth
gdown --id 1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a -O examples/ckpt/ckpt_116_i2i_comb.pth
gdown --id 18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI -O examples/dump/emb.pickle
cp ../scripts/train_image_translation.py src/approaches/train_image_translation.py
cd ..
mv MakeItTalk src/utils
