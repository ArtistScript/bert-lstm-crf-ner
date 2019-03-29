#!/bin/bash

#########################################################################
# File Name: run.sh
# Create Time: 2018-11-19 18:38:52
# Author: liuhong
# Last Modified: 2018-11-20 19:05:32
# Description: 
#########################################################################
#echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > ./input.txt
echo '红色' > ./input.txt
echo '大红色' >> ./input.txt
python  sentence2veclib.py ./sentence2vec.json ./output.json
