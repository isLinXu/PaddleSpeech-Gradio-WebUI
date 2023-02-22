#!/bin/bash
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple

pip install -r requirements.txt

apt-get install libsndfile1 -y



# 创建工作目录
mkdir -p ./work/workspace_asr
cd ./work/workspace_asr

# 检测模型是否存在
test -f transformer.model.tar.gz || wget -nc https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/transformer.model.tar.gz

test -d conf || tar xzvf transformer.model.tar.gz

# 示例音频文件
test -f ./data/demo_01_03.wav || wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/demo_01_03.wav -P ./data/

# 返回工作目录
cd ../../


