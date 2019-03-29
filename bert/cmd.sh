#!/bin/bash

#########################################################################
# File Name: run.sh
# Create Time: 2018-11-19 18:38:52
# Author: liuhong
# Last Modified: 2018-11-19 18:40:30
# Description: 
#########################################################################
#echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > ./input.txt
echo '红色' > ./input.txt
echo '大红色' >> ./input.txt
BERT_BASE_DIR=$1
python  sentence2vec.py \
  --input_file=./input.txt \
  --output_file=./output.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --feature_pooling_layer=-2 \
  --max_seq_length=128 \
  --batch_size=8
