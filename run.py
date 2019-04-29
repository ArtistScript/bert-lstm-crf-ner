#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
reference from :zhoukaiyin/

@Author:Macan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#sys.path.append('/home/idm/dzt/models/nlp_model/bert-lstm-crf-ner/')
import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle

from train import tf_metrics
from bert import modeling
from bert import optimization
from bert import tokenization

# import

from train.models import create_model, InputFeatures, InputExample

__version__ = '0.1.0'

__all__ = ['__version__', 'DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           'model_fn_builder', 'train']



if __name__=='__main__':
    import os
    from train.train_helper import get_args_parser
    from train.bert_lstm_ner_cg_estimator import train

    args = get_args_parser()
    # print(args)
    operation_sys="linux"
    if operation_sys=="windows":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        args.task_name="NER"
        args.do_train=True
        args.do_eval=True
        args.do_predict=True
        args.data_dir="D:/project/python_project/bert-lstm-crf-ner/data"
        args.vocab_file="D:/project/python_project/bert-lstm-crf-ner/bert\chinese_L-12_H-768_A-12/vocab.txt"
        args.bert_config_file="D:/project/python_project/bert-lstm-crf-ner/bert/chinese_L-12_H-768_A-12/bert_config.json"
        args.init_checkpoint="D:/project/python_project/bert-lstm-crf-ner/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        args.max_seq_length=128
        args.train_batch_size=32
        args.batch_size = 32
        args.learning_rate=2e-5
        args.num_train_epochs=3.0
        args.output_dir="D:/project/python_project/bert-lstm-crf-ner/output"
    elif operation_sys=="linux":
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        args.device_map="1,2,3"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        args.task_name = "NER"
        args.do_train = True
        args.do_eval = True
        args.do_predict = True
        args.data_dir = "/home/idm/dzt/models/nlp_model/bert-lstm-crf-ner/data/"
        args.vocab_file = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/vocab.txt"
        args.bert_config_file = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/bert_config.json"
        args.init_checkpoint = "/home/idm/dzt/kaola-ner/chinese_L-12_H-768_A-12/bert_model.ckpt"
        args.max_seq_length = 128
        args.train_batch_size = 32
        args.batch_size = 32  #这个是用于设置batch_size的
        args.learning_rate = 2e-5
        args.num_train_epochs = 3.0
        args.output_dir = "./output"
        # args.dropout_rate=0.1  #ner模型的训练dropout率，在中间层可以设置大一点，比如0.5
    else:
        print("Please input the ")
    if True:
        import sys

        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    train(args=args)
