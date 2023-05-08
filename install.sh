#!/bin/bash

cd src/utils


git clone https://github.com/yzhou359/MakeItTalk
cd MakeItTalk
mkdir examples/dump
mkdir examples/ckpt
gdown --id 1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x -O examples/ckpt/ckpt_autovc.pth
gdown --id 1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp -O examples/ckpt/ckpt_content_branch.pth
gdown --id 1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu -O examples/ckpt/ckpt_speaker_branch.pth
gdown --id 1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a -O examples/ckpt/ckpt_116_i2i_comb.pth
gdown --id 18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI -O examples/dump/emb.pickle
cp ../../scripts/train_image_translation.py src/approaches/train_image_translation.py
cp ../../scripts/models.py thirdparty/AdaptiveWingLoss/core/models.py
cd ..
